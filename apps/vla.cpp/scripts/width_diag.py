"""Offline width-pruning diagnostic: remove FFN neurons from the LM and/or the action expert
and measure how far the actions move (same metrics and samples as prune_diag.py).

Width pruning here = removing neurons of the SwiGLU FFN (down_proj(silu(gate(x)) * up(x))):
neuron j of a layer is row j of gate_proj and up_proj plus column j of down_proj. The
residual width (LM 960, expert 480) is untouched, so any layer can be cut. vla.cpp stores
ONE FFN size for all LM layers (smolvla.intermediate) and one for all expert layers
(smolvla.expert_inter), so every layer of a module loses the same number of neurons;
which neurons go is chosen per layer.

In PyTorch the neurons are removed by zeroing their down_proj columns: a neuron whose
output column is zero contributes nothing, which is exactly the same as deleting it.

Neuron scores (lowest score is removed first; these definitions are our own choice):
  act   Wanda-style, aggregated per neuron: RMS of the neuron's activation (input of
        down_proj) on calibration tokens  x  L2 norm of its down_proj column.
        LM tokens = non-padding prefix tokens; expert tokens = the 50 action tokens at
        every denoising step.
  mag   weights only: ||gate row|| x ||up row|| x ||down column||.
  rand  random order per layer (seed --seed).
Scores are computed on the calibration samples that are NOT used for evaluation, so the
evaluation does not reuse the samples the choice was fitted on.

Smoke test 2026-11-10 (10 eval samples, 60 scoring samples), removing 25% act-lowest
neurons in ONE layer only: LM layers 0 and 3 alone already give rel ~1 (actions wrecked),
LM layers 24-31 almost nothing (layer 31: 0.000, its FFN output is unused); every expert
layer alone 0.01-0.03. LM layer 3 has one neuron with act score 173 (median 0.52), a
"massive activation". Because vla.cpp forces the same FFN size on every LM layer, the
fragile early LM layers make uniform LM width pruning break the model, and the LM curve
is not monotonic (a broken model gives arbitrary actions).

Usage (inside WSL, LIBERO venv, GPU free = no vla-server running):
  python width_diag.py --calib ~/profile/calib --ckpt ~/models/smolvla_libero_hf \
      --out results/prune/width
Writes width_dose.csv, scores.npz (every score, for building pruned checkpoints later).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from prune_diag import load_policy
from replay_calib import calib_files, select_rows

ARM, GRIP, ADIM = slice(0, 6), 6, 7


class Inputs:
    """Builds the model inputs of one calibration sample (same as prune_diag.py)."""

    def __init__(self, model, stats, device):
        self.chunk, self.adim_pad = model.config.chunk_size, model.config.max_action_dim
        self.stats, self.device = stats, device

    def __call__(self, s):
        dev, eps = self.device, 1e-8
        imgs = torch.from_numpy(s["images"]).float().div(255)
        images = [im.permute(2, 0, 1)[None].to(dev) * 2 - 1 for im in imgs]
        img_masks = [torch.ones(1, dtype=torch.bool, device=dev) for _ in images]
        lang = torch.from_numpy(s["tokens"]).long()[None].to(dev)
        lang_mask = (torch.arange(lang.shape[1]) < int(s["n_tokens"]))[None].to(dev)
        state = torch.from_numpy(s["state"]).float().clone()        # clone: never modify the sample
        sd = self.stats["s_mean"].numel()
        state[:sd] = (state[:sd] - self.stats["s_mean"]) / (self.stats["s_std"] + eps)
        noise = torch.from_numpy(s["noise"]).float().view(1, self.chunk, self.adim_pad).to(dev)
        return images, img_masks, lang, lang_mask, state[None].to(dev), noise


class NeuronZeroer:
    """Zero down_proj columns (= remove FFN neurons) and restore them again."""

    def __init__(self, mlps):
        self.mlps = mlps                                   # {"lm": [mlp...], "expert": [mlp...]}
        self.backup = {}                                   # (module, layer) -> CPU copy of down_proj.weight

    def set(self, drop):                                   # drop: {(module, layer): LongTensor of neurons}
        with torch.no_grad():
            for (mod, l), w in self.backup.items():
                self.mlps[mod][l].down_proj.weight.copy_(w)
            self.backup = {}
            for (mod, l), idx in drop.items():
                W = self.mlps[mod][l].down_proj.weight     # [hidden, intermediate]
                self.backup[(mod, l)] = W.detach().cpu().clone()
                W[:, idx.to(W.device)] = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=10, help="evaluation samples per task (as prune_diag.py)")
    ap.add_argument("--max-score-samples", type=int, default=0, help="cap on scoring samples, 0 = all the rest")
    ap.add_argument("--ratios", default="0.05,0.125,0.25,0.5", help="fraction of FFN neurons removed per layer")
    ap.add_argument("--targets", default="lm,expert", help="lm, expert and/or both")
    ap.add_argument("--criteria", default="act,mag,rand")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    policy, stats = load_policy(a.ckpt.expanduser(), a.device)
    model = policy.model
    vwe = model.vlm_with_expert
    mlps = {"lm": [l.mlp for l in vwe.get_vlm_model().text_model.layers],
            "expert": [l.mlp for l in vwe.lm_expert.layers]}
    zeroer = NeuronZeroer(mlps)
    make = Inputs(model, stats, a.device)

    files = calib_files(a.calib)
    data = [np.load(f) for f in files]
    keys = ("images", "tokens", "n_tokens", "state", "noise", "action")
    take = lambda t, r: {k: data[t][k][r] for k in keys}
    ev_sel = select_rows(files, a.per_task)
    ev_set = set(ev_sel)
    sc_sel = [(t, r) for t in range(len(files)) for r in range(len(data[t]["n_tokens"])) if (t, r) not in ev_set]
    if a.max_score_samples:
        sc_sel = sc_sel[::max(1, len(sc_sel) // a.max_score_samples)][:a.max_score_samples]
    ev = [take(t, r) for t, r in ev_sel]
    print(f"{len(ev)} evaluation samples, {len(sc_sel)} scoring samples", flush=True)

    # ---- 1. activation statistics: sum of h^2 per neuron, h = input of down_proj
    sumsq = {m: [None] * len(v) for m, v in mlps.items()}
    count = {m: [0] * len(v) for m, v in mlps.items()}
    cur = {"lang_mask": None}

    def hook(mod, l):
        def f(_m, args):
            h = args[0][0].detach()                                    # [tokens, intermediate]
            if mod == "lm":                                            # prefix = [images | prompt (48) | state]
                lm_ = cur["lang_mask"]
                keep = torch.cat([torch.ones(h.shape[0] - lm_.numel() - 1, dtype=torch.bool), lm_,
                                  torch.ones(1, dtype=torch.bool)]).to(h.device)
                h = h[keep]
            hh = (h.double() ** 2).sum(0).cpu()
            sumsq[mod][l] = hh if sumsq[mod][l] is None else sumsq[mod][l] + hh
            count[mod][l] += h.shape[0]
        return f

    handles = [mlp.down_proj.register_forward_pre_hook(hook(m, l)) for m, v in mlps.items() for l, mlp in enumerate(v)]
    t0 = time.time()
    for i, (t, r) in enumerate(sc_sel):
        x = make(take(t, r))
        cur["lang_mask"] = x[3][0].cpu()
        with torch.no_grad():
            model.sample_actions(*x[:5], noise=x[5])
        if (i + 1) % 40 == 0:
            print(f"scoring {i + 1}/{len(sc_sel)}, {time.time() - t0:.0f}s", flush=True)
    for h in handles:
        h.remove()

    # ---- 2. neuron scores per (module, layer)
    rng = np.random.default_rng(a.seed)
    scores = {}
    for m, v in mlps.items():
        for l, mlp in enumerate(v):
            down = mlp.down_proj.weight.detach().float().cpu()          # [hidden, inter]
            col = down.norm(dim=0)
            rms = (sumsq[m][l] / count[m][l]).sqrt().float()
            scores[("act", m, l)] = (rms * col).numpy()
            scores[("mag", m, l)] = (mlp.gate_proj.weight.detach().float().cpu().norm(dim=1)
                                     * mlp.up_proj.weight.detach().float().cpu().norm(dim=1) * col).numpy()
            scores[("rand", m, l)] = rng.random(col.numel())
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "scores.npz", **{f"{c}_{m}_{l}": s for (c, m, l), s in scores.items()})
    print(f"scores done ({time.time() - t0:.0f}s): LM {len(mlps['lm'])} x {scores[('act', 'lm', 0)].size}, "
          f"expert {len(mlps['expert'])} x {scores[('act', 'expert', 0)].size} neurons", flush=True)

    # ---- 3. evaluate every (criterion, target, ratio)
    def run():
        acts = []
        for s in ev:
            x = make(s)
            with torch.no_grad():
                y = model.sample_actions(*x[:5], noise=x[5])
            acts.append((y[0, :, :ADIM].float().cpu() * (stats["a_std"] + 1e-8) + stats["a_mean"]).numpy())
        return np.stack(acts)

    def compare(p, ref):
        arm0 = np.abs(p[:, 0, ARM] - ref[:, 0, ARM]).mean(1)
        return {"arm_t0": arm0.mean(), "arm_p90": np.percentile(arm0, 90),
                "arm_all": np.abs(p[..., ARM] - ref[..., ARM]).mean(),
                "grip_t0": np.mean(np.sign(p[:, 0, GRIP]) != np.sign(ref[:, 0, GRIP]))}

    zeroer.set({})
    full = run()
    recorded = np.stack([s["action"][:, :ADIM] for s in ev])
    scale = np.abs(full[:, 0, ARM]).mean()
    print(f"full model: torch vs recorded server arm_t0 {compare(full, recorded)['arm_t0']:.2e}; "
          f"mean |arm action| at t0 {scale:.3e}", flush=True)

    mods = {"lm": ["lm"], "expert": ["expert"], "both": ["lm", "expert"]}
    rows = []
    for crit in a.criteria.split(","):
        for tgt in a.targets.split(","):
            for ratio in (float(x) for x in a.ratios.split(",")):
                t1 = time.time()
                drop = {}
                for m in mods[tgt]:
                    for l in range(len(mlps[m])):
                        s = scores[(crit, m, l)]
                        drop[(m, l)] = torch.from_numpy(np.argsort(s, kind="stable")[:int(round(ratio * s.size))].copy())
                zeroer.set(drop)
                res = compare(run(), full)
                zeroer.set({})
                inter = {m: scores[(crit, m, 0)].size for m in mods[tgt]}
                kept = " ".join(f"{m}:{n - int(round(ratio * n))}/{n}" for m, n in inter.items())
                rows.append({"criterion": crit, "target": tgt, "ratio": ratio, "ffn_kept": kept,
                             **{k: f"{v:.4e}" if k != "grip_t0" else f"{v:.3f}" for k, v in res.items()},
                             "arm_t0_rel": f"{res['arm_t0'] / scale:.3f}", "seconds": f"{time.time() - t1:.0f}"})
                print(f"{crit:4s} {tgt:6s} {ratio:5.3f} [{kept}] arm_t0 rel {res['arm_t0'] / scale:.3f} "
                      f"p90 {res['arm_p90']:.3e} grip_t0 {res['grip_t0']:.3f}  {rows[-1]['seconds']}s", flush=True)

    again = run()                                          # every zeroed column must have been restored
    print(f"restore check: max |full after all configs - full before| = {np.abs(again - full).max():.2e}")

    with open(out / "width_dose.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"total {time.time() - t0:.0f}s, wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
