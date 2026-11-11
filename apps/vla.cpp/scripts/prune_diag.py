"""Offline pruning diagnostic: how far do the actions move when layer pairs are removed?

Training-free pruning screen round 1 (8 of 32 layer pairs removed) gave ~0 % success for
the cka, last and random choices. This script measures the damage without the simulator,
on the recorded calibration requests (collect_calib.py), for every criterion and every
number of removed pairs k:

  1. Dose-response. The SmolVLA PyTorch model (lerobot) is pruned in memory with the same
     effect as prune_smolvla.py on disk (LM layer i and expert layer i removed together;
     done by zeroing their output projections, see Pruner) and run on the same inputs +
     the same recorded noise. Each pruned
     action chunk is compared with the unpruned PyTorch chunk:
       arm_t0    mean |delta| of the 6 arm dims at chunk step 0 (the only step the LIBERO
                 client executes, n_action_steps=1)
       arm_p90   90th percentile over samples of that per-sample value (tails)
       arm_all   same as arm_t0 but over all 50 chunk steps
       grip_t0   fraction of samples whose gripper command changes sign at step 0
  2. Bug check (--server). Actions that vla-server returned for a pruned GGUF
     (replay_calib.py) are compared with the PyTorch model pruned the same way. If they
     agree about as well as the unpruned pair does, the GGUF pipeline (renumbering,
     conversion, loading) is not the cause of the failures.

The drop lists come from cka_layers.py's outputs (--cka-dir): cka and cos pairs are
re-derived from pair_scores.csv with the same greedy rule and must match drop_lists.txt
where it has an entry; last = the last 2k layers; rand uses drop_lists.txt when it has
that k (so rand_k4 is the screened model) and otherwise a new draw (marked "new draw").

Usage (inside WSL, LIBERO venv, GPU free = no vla-server running):
  python prune_diag.py --calib ~/profile/calib --ckpt ~/models/smolvla_libero_hf \
      --cka-dir results/prune/cka --out results/prune/diag \
      [--server full=results/prune/diag/server_full.npz --server cka_k4=results/prune/diag/server_cka_k4.npz]
Writes dose.csv, actions.npz (every config's [N, 50, 7] actions) and, with --server, check.txt.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cka_layers import greedy_pairs
from prune_smolvla import check_types, parse_drop
from replay_calib import calib_files, select_rows

ARM, GRIP, ADIM = slice(0, 6), 6, 7


def load_policy(ckpt, device):
    """lerobot SmolVLAPolicy from the HF checkpoint + (de)normalization stats.
    Same loading as cka_layers.py (tokenizer-only processor stand-in, no SmolVLM2 download)."""
    from types import SimpleNamespace
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla import smolvlm_with_expert
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    class _TokenizerOnlyProcessor:
        @staticmethod
        def from_pretrained(model_id, *_args, **_kwargs):
            return SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(model_id))
    smolvlm_with_expert.AutoProcessor = _TokenizerOnlyProcessor

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.load_vlm_weights = False
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg).to(device, torch.float32).eval()
    st = load_file(str(ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    at = load_file(str(ckpt / "policy_postprocessor_step_1_unnormalizer_processor.safetensors"))
    stats = {k: v.float() for k, v in (("s_mean", st["observation.state.mean"]),
                                        ("s_std", st["observation.state.std"]),
                                        ("a_mean", at["action.mean"]), ("a_std", at["action.std"]))}
    return policy, stats


class Pruner:
    """Remove layer pairs by zeroing their residual-branch outputs (self_attn.o_proj and
    mlp.down_proj of LM layer i and expert layer i). A zeroed layer passes the residual
    stream through unchanged, and the K/V it still produces are only read by its own
    (zeroed) expert twin, so this equals deleting the pair. Checked 2026-11-10 on 10
    calibration samples: vla.cpp drop_30_31.gguf == vla.cpp 32-layer GGUF with layers 30,31
    zeroed (diff 0.0), and PyTorch zeroed == vla.cpp zeroed (arm diff 2.6e-4, same level as
    the unpruned pair). Unlike deleting from the ModuleLists, nothing in lerobot that counts
    layers (num_vlm_layers, get_model_layers) has to be kept in sync."""

    def __init__(self, vwe):
        self.vwe = vwe
        txt, exp = vwe.get_vlm_model().text_model, vwe.lm_expert
        self.n = len(txt.layers)
        assert len(exp.layers) == self.n
        self.weights = [[l.self_attn.o_proj.weight, l.mlp.down_proj.weight]
                        for l in (*txt.layers, *exp.layers)]          # LM i at [i], expert i at [n + i]
        self.backup = {}                                            # index -> CPU copies of zeroed weights
        self.drop = []

    def set(self, drop):
        kept = [i for i in range(self.n) if i not in set(drop)]
        probs = check_types(kept, self.vwe.self_attn_every_n_layers)   # only sets prune_smolvla.py can build
        if probs:
            raise SystemExit("SA/CA type would change: " + "; ".join(probs[:3]))
        with torch.no_grad():
            for j, saved in self.backup.items():                       # restore the previous config
                for w, s in zip(self.weights[j], saved):
                    w.copy_(s)
            self.backup = {}
            for i in drop:
                for j in (i, self.n + i):
                    self.backup[j] = [w.detach().cpu().clone() for w in self.weights[j]]
                    for w in self.weights[j]:
                        w.zero_()
        self.drop = list(drop)


def drop_configs(cka_dir, ks, criteria, n_layers, seed):
    """-> list of (name, criterion, k, drop list, note)."""
    with open(cka_dir / "pair_scores.csv") as f:
        rows = list(csv.DictReader(f))
    score = {"cka": [float(r["cka_mean"]) for r in rows], "cos": [float(r["cos_mean"]) for r in rows]}
    listed = {}
    for line in (cka_dir / "drop_lists.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, spec, _method = line.split("|")
            listed[name] = parse_drop(spec)

    out = []
    for k in ks:
        for c in criteria:
            name, note = f"{c}_k{k}", ""
            if c in score:
                drop = sorted(j for i in greedy_pairs(score[c], k) for j in (i, i + 1))
                if name in listed and listed[name] != drop:
                    raise SystemExit(f"{name}: re-derived {drop} != drop_lists.txt {listed[name]}")
            elif c == "last":
                drop = list(range(n_layers - 2 * k, n_layers))
            elif c == "rand":
                if name in listed:
                    drop = listed[name]
                else:
                    rng = random.Random(seed + 1000 * k)
                    drop = sorted(j for i in greedy_pairs([rng.random() for _ in range(n_layers - 1)], k)
                                  for j in (i, i + 1))
                    note = "new draw"
            else:
                raise SystemExit(f"unknown criterion {c}")
            out.append((name, c, k, drop, note))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cka-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=10, help="samples per task file (same rule as replay_calib.py)")
    ap.add_argument("--ks", default="1,2,3,4", help="numbers of removed layer pairs")
    ap.add_argument("--criteria", default="cka,cos,last,rand")
    ap.add_argument("--seed", type=int, default=0, help="random baseline seed (as in cka_layers.py)")
    ap.add_argument("--server", action="append", default=[], metavar="NAME=NPZ",
                    help="vla-server actions from replay_calib.py for config NAME (or 'full'); repeatable")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    policy, stats = load_policy(a.ckpt.expanduser(), a.device)
    model = policy.model
    pruner = Pruner(model.vlm_with_expert)
    chunk, adim_pad = model.config.chunk_size, model.config.max_action_dim
    eps = 1e-8

    files = calib_files(a.calib)
    sel = select_rows(files, a.per_task)
    data = [np.load(f) for f in files]
    samples = []                                       # preloaded once, reused by every config
    for t, r in sel:
        d = data[t]
        samples.append({k: d[k][r] for k in ("images", "tokens", "n_tokens", "state", "noise", "action")})
    print(f"{len(samples)} samples from {len(files)} task files", flush=True)

    def run(drop):
        """Actions [N, 50, 7] (unnormalized) of the model with `drop` removed."""
        pruner.set(drop)
        acts = []
        for s in samples:
            imgs = torch.from_numpy(s["images"]).float().div(255)
            images = [im.permute(2, 0, 1)[None].to(a.device) * 2 - 1 for im in imgs]
            img_masks = [torch.ones(1, dtype=torch.bool, device=a.device) for _ in images]
            lang = torch.from_numpy(s["tokens"]).long()[None].to(a.device)
            lang_mask = (torch.arange(lang.shape[1]) < int(s["n_tokens"]))[None].to(a.device)
            state = torch.from_numpy(s["state"]).float().clone()    # .float() of float32 shares memory:
            sd = stats["s_mean"].numel()                               # without clone the preloaded sample is normalized again every run
            state[:sd] = (state[:sd] - stats["s_mean"]) / (stats["s_std"] + eps)
            noise = torch.from_numpy(s["noise"]).float().view(1, chunk, adim_pad).to(a.device)
            with torch.no_grad():
                x = model.sample_actions(images, img_masks, lang, lang_mask, state[None].to(a.device), noise=noise)
            act = x[0, :, :ADIM].float().cpu() * (stats["a_std"] + eps) + stats["a_mean"]
            acts.append(act.numpy())
        pruner.set([])
        return np.stack(acts)

    def compare(p, ref):
        arm0 = np.abs(p[:, 0, ARM] - ref[:, 0, ARM]).mean(1)            # per sample
        return {"arm_t0": arm0.mean(), "arm_p90": np.percentile(arm0, 90),
                "arm_all": np.abs(p[..., ARM] - ref[..., ARM]).mean(),
                "grip_t0": np.mean(np.sign(p[:, 0, GRIP]) != np.sign(ref[:, 0, GRIP]))}

    t0 = time.time()
    full = run([])
    recorded = np.stack([s["action"][:, :ADIM] for s in samples])
    sanity = compare(full, recorded)
    scale = np.abs(full[:, 0, ARM]).mean()
    print(f"full model: {time.time() - t0:.0f}s; torch vs recorded server: arm_t0 {sanity['arm_t0']:.2e} "
          f"grip_t0 {sanity['grip_t0']:.3f}; mean |arm action| at t0 {scale:.3e}", flush=True)

    configs = drop_configs(a.cka_dir.expanduser(), [int(k) for k in a.ks.split(",")],
                           a.criteria.split(","), pruner.n, a.seed)
    acts = {"full": full}
    cache = {}                                          # same drop list -> computed once
    rows = []
    for name, crit, k, drop, note in configs:
        t1 = time.time()
        key = tuple(drop)
        if key not in cache:
            cache[key] = run(drop)
        acts[name] = cache[key]
        m = compare(cache[key], full)
        rows.append({"name": name, "criterion": crit, "k": k, "layers_dropped": len(drop),
                     "drop": " ".join(map(str, drop)), "note": note,
                     **{x: f"{m[x]:.4e}" if x != "grip_t0" else f"{m[x]:.3f}" for x in m},
                     "arm_t0_rel": f"{m['arm_t0'] / scale:.3f}", "seconds": f"{time.time() - t1:.0f}"})
        print(f"{name:8s} drop [{rows[-1]['drop']}] arm_t0 {m['arm_t0']:.3e} (rel {m['arm_t0'] / scale:.3f}) "
              f"p90 {m['arm_p90']:.3e} grip_t0 {m['grip_t0']:.3f}  {note} {rows[-1]['seconds']}s", flush=True)

    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dose.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(out / "actions.npz", task=np.array([t for t, _ in sel]), row=np.array([r for _, r in sel]), **acts)

    if a.server:
        lines = [f"torch full vs recorded server (float images): arm_t0 {sanity['arm_t0']:.3e} "
                 f"grip_t0 {sanity['grip_t0']:.3f}"]
        for item in a.server:
            name, _, path = item.partition("=")
            z = np.load(Path(path).expanduser())
            if not (np.array_equal(z["task"], [t for t, _ in sel]) and np.array_equal(z["row"], [r for _, r in sel])):
                raise SystemExit(f"{path}: samples differ from this run (use the same --per-task)")
            if name not in acts:
                raise SystemExit(f"--server {name}: no such config in this run")
            srv = z["action"][..., :ADIM]
            m, m_full = compare(srv, acts[name]), compare(acts[name], full)
            lines.append(f"{name}: vla.cpp vs torch (same pruning) arm_t0 {m['arm_t0']:.3e} arm_all {m['arm_all']:.3e} "
                         f"grip_t0 {m['grip_t0']:.3f}   | torch pruned vs torch full arm_t0 {m_full['arm_t0']:.3e}")
        (out / "check.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    print(f"total {time.time() - t0:.0f}s, wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
