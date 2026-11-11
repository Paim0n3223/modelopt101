"""Training-free weight compensation for pruned SmolVLA (least squares on calibration data),
measured with the same samples and metric as prune_diag.py / width_diag.py.

Only VALUES of existing weights change (shapes stay), so a compensated checkpoint still runs
on the unmodified vla.cpp.

Depth (layer pairs removed as in prune_diag.py). Removed layers form maximal contiguous runs
[a..b]. Per module (LM, expert), the residual stream before layer b+1 used to be
h(b+1) = h(a) + delta and is now h(a). The kept layer a-1 gets a linear correction on its
down_proj:  W(a-1) += M,  M = argmin E|| delta - M u ||^2 + lam ||M||^2,  u = input of
down_proj(a-1). So   M = E[delta u^T] (E[u u^T] + lam I)^-1.
(Our own design; same spirit as the least-squares reconstruction papers in NOTES §13.)
Notes: delta and u come from the UNPRUNED model (one pass, no sequential refit). LM runs
that end at the last layer are not compensated: nothing kept reads the LM stream after them.

Width (expert FFN neurons removed as in width_diag.py, criterion `act`). Per expert layer,
the kept down_proj columns are refitted so the layer output matches the unpruned one:
  W'_K = W C[:, K] (C[K, K] + lam I)^-1,  C = E[h h^T],  h = input of down_proj.

lam = alpha * mean(diag of the Gram matrix).

Smoke test 2026-11-10 (10 eval samples, 80 fitting samples, --nocomp for a same-sample
reference): cka_k2 0.415 -> 0.258, last_k2 0.480 -> 0.269, expert width 50% 0.578 -> 0.193;
alpha 1e-4 .. 1e-2 changed depth results by < 0.01, width preferred small alpha (0.19 vs 0.22).
Fit on the calibration samples NOT used for evaluation; evaluate on the same 100 samples.

Usage (inside WSL, LIBERO venv, GPU free):
  python compensate_diag.py --calib ~/profile/calib --ckpt ~/models/smolvla_libero_hf \
      --cka-dir results/prune/cka --width-scores results/prune/width/scores.npz \
      --out results/prune/comp
Writes comp_dose.csv (with the uncompensated rel from dose.csv / width_dose.csv beside it).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from prune_diag import Pruner, drop_configs, load_policy
from replay_calib import calib_files, select_rows
from width_diag import Inputs

ARM, GRIP, ADIM = slice(0, 6), 6, 7


def runs_of(drop):
    """Sorted layer indices -> maximal contiguous runs [(a, b), ...]."""
    out = []
    for i in sorted(drop):
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


class Patcher:
    """Overwrite down_proj weights in place and restore them afterwards."""

    def __init__(self):
        self.backup = []

    def set(self, w, new):
        with torch.no_grad():
            self.backup.append((w, w.detach().cpu().clone()))
            w.copy_(new.to(w.device, w.dtype))

    def restore(self):
        with torch.no_grad():
            for w, old in reversed(self.backup):
                w.copy_(old.to(w.device))
        self.backup = []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cka-dir", type=Path, required=True)
    ap.add_argument("--width-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--max-fit-samples", type=int, default=0, help="0 = all non-evaluation samples")
    ap.add_argument("--ks", default="1,2,3,4", help="depth: numbers of removed pairs ('' = none)")
    ap.add_argument("--criteria", default="cka,cos,last,rand")
    ap.add_argument("--width-ratios", default="0.25,0.5", help="expert FFN, criterion act ('' = none)")
    ap.add_argument("--alphas", default="0.001", help="ridge strength(s), relative to mean diag of the Gram")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of config names to run (e.g. cka_k4,wexp_act0.5); '' = all")
    ap.add_argument("--save", default="", help="comma list of config names whose compensated 32-layer "
                    "checkpoint is written to --save-dir/<name>_comp32 (then prune it as usual)")
    ap.add_argument("--save-dir", type=Path, default=Path.home() / "models/smolvla-comp")
    ap.add_argument("--nocomp", action="store_true",
                    help="also evaluate every config WITHOUT compensation in this run (needed when the "
                         "evaluation samples differ from dose.csv / width_dose.csv, e.g. --per-task != 10)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    policy, stats = load_policy(a.ckpt.expanduser(), dev)
    model = policy.model
    vwe = model.vlm_with_expert
    txt, exp = vwe.get_vlm_model().text_model, vwe.lm_expert
    layers = {"lm": list(txt.layers), "expert": list(exp.layers)}
    n = len(layers["lm"])
    pruner, patcher, make = Pruner(vwe), Patcher(), Inputs(model, stats, dev)

    depth = drop_configs(a.cka_dir.expanduser(), [int(k) for k in a.ks.split(",") if k],
                         a.criteria.split(","), n, a.seed) if a.ks else []
    ratios = [float(r) for r in a.width_ratios.split(",") if r]
    alphas = [float(x) for x in a.alphas.split(",")]
    only = {s for s in a.only.split(",") if s}
    to_save = {s for s in a.save.split(",") if s}
    if only:
        depth = [c for c in depth if c[0] in only]
        ratios = [r for r in ratios if f"wexp_act{r}" in only]
    known = {c[0] for c in depth} | {f"wexp_act{r}" for r in ratios}
    if to_save - known:
        raise SystemExit(f"--save names not among the configs of this run: {sorted(to_save - known)}")
    if to_save and len(alphas) != 1:
        raise SystemExit("--save needs exactly one alpha")

    # ---- what has to be measured
    runs = {}                                            # (mod, a, b) needed for depth compensation
    for _name, _c, _k, drop, _note in depth:
        for ra, rb in runs_of(drop):
            if ra == 0:
                continue
            runs[("expert", ra, rb)] = None
            if rb < n - 1:
                runs[("lm", ra, rb)] = None
    gram_layers = {("lm", ra - 1) for (m, ra, rb) in runs if m == "lm"}
    gram_layers |= {("expert", l) for l in range(n)} if ratios else set()
    gram_layers |= {("expert", ra - 1) for (m, ra, rb) in runs if m == "expert"}
    state_idx = {m: sorted({ra for (mm, ra, rb) in runs if mm == m} | {rb + 1 for (mm, ra, rb) in runs if mm == m})
                 for m in ("lm", "expert")}
    print(f"{len(depth)} depth configs, {len(runs)} runs to compensate, {len(gram_layers)} Gram matrices, "
          f"width ratios {ratios}, alphas {alphas}", flush=True)

    # ---- accumulators: GPU float32, flushed to CPU float64
    G = {k: None for k in gram_layers}
    X = {k: None for k in runs}
    G64 = {k: 0 for k in gram_layers}
    X64 = {k: 0 for k in runs}
    cnt = {"lm": 0, "expert": 0}
    cur = {"lang_mask": None, "lm": {}, "expert": {}, "u": {}}

    def add(d, k, v):
        d[k] = v if d[k] is None else d[k] + v

    def rows(mod, t):                                    # [1, T, D] -> [rows, D] (LM: drop prompt padding)
        t = t[0]
        if mod == "lm":
            lm_ = cur["lang_mask"]
            keep = torch.cat([torch.ones(t.shape[0] - lm_.numel() - 1, dtype=torch.bool), lm_,
                              torch.ones(1, dtype=torch.bool)]).to(t.device)
            t = t[keep]
        return t.float()

    def state_hook(mod, j):
        def f(_m, args):
            cur[mod][j] = rows(mod, args[0].detach())
        return f

    def down_hook(mod, l):
        def f(_m, args):
            cur["u"][(mod, l)] = rows(mod, args[0].detach())
        return f

    def end_hook(mod):                                   # end of one pass of `mod` (final norm input)
        def f(_m, args):
            cur[mod][n] = rows(mod, args[0].detach())
            for k in gram_layers:
                if k[0] == mod:
                    u = cur["u"][k]
                    add(G, k, u.T @ u)
            for k in runs:
                if k[0] == mod:
                    _, ra, rb = k
                    delta = cur[mod][rb + 1] - cur[mod][ra]
                    add(X, k, delta.T @ cur["u"][(mod, ra - 1)])
            cnt[mod] += cur[mod][n].shape[0]
            cur[mod].clear()
            cur["u"] = {kk: v for kk, v in cur["u"].items() if kk[0] != mod}
        return f

    handles = []
    for mod, mdl in (("lm", txt), ("expert", exp)):
        for j in state_idx[mod]:
            if j < n:
                handles.append(layers[mod][j].input_layernorm.register_forward_pre_hook(state_hook(mod, j)))
        handles.append(mdl.norm.register_forward_pre_hook(end_hook(mod)))
    for mod, l in gram_layers:
        handles.append(layers[mod][l].mlp.down_proj.register_forward_pre_hook(down_hook(mod, l)))

    files = calib_files(a.calib)
    data = [np.load(f) for f in files]
    keys = ("images", "tokens", "n_tokens", "state", "noise", "action")
    take = lambda t, r: {k: data[t][k][r] for k in keys}
    ev_sel = select_rows(files, a.per_task)
    ev_set = set(ev_sel)
    fit_sel = [(t, r) for t in range(len(files)) for r in range(len(data[t]["n_tokens"])) if (t, r) not in ev_set]
    if a.max_fit_samples:
        fit_sel = fit_sel[::max(1, len(fit_sel) // a.max_fit_samples)][:a.max_fit_samples]
    ev = [take(t, r) for t, r in ev_sel]
    print(f"{len(ev)} evaluation samples, {len(fit_sel)} fitting samples", flush=True)

    def flush():
        for d, d64 in ((G, G64), (X, X64)):
            for k, v in d.items():
                if v is not None:
                    d64[k] = d64[k] + v.double().cpu()
                    d[k] = None

    t0 = time.time()
    for i, (t, r) in enumerate(fit_sel):
        x = make(take(t, r))
        cur["lang_mask"] = x[3][0].cpu()
        with torch.no_grad():
            model.sample_actions(*x[:5], noise=x[5])
        if (i + 1) % 16 == 0:
            flush()
        if (i + 1) % 40 == 0:
            print(f"fitting pass {i + 1}/{len(fit_sel)}, {time.time() - t0:.0f}s", flush=True)
    flush()
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()
    print(f"statistics done ({time.time() - t0:.0f}s): rows lm {cnt['lm']}, expert {cnt['expert']}", flush=True)

    def ridge_solve(Gm, B, alpha):
        """B (G + lam I)^-1 for symmetric G, lam = alpha * mean diag(G)."""
        lam = alpha * float(torch.diagonal(Gm).mean())
        A = Gm + lam * torch.eye(Gm.shape[0], dtype=Gm.dtype)
        return torch.linalg.solve(A, B.T).T

    # ---- evaluation
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

    full = run()
    scale = np.abs(full[:, 0, ARM]).mean()
    print(f"full model: mean |arm action| at t0 {scale:.3e}", flush=True)

    base = {}
    dose = a.cka_dir.expanduser().parent / "diag" / "dose.csv"
    if dose.exists():
        base.update({("depth", r["name"]): r["arm_t0_rel"] for r in csv.DictReader(open(dose))})
    wd = a.width_scores.expanduser().parent / "width_dose.csv"
    if wd.exists():
        base.update({("width", f"wexp_act{float(r['ratio'])}"): r["arm_t0_rel"]
                     for r in csv.DictReader(open(wd)) if r["criterion"] == "act" and r["target"] == "expert"})

    if a.nocomp or a.per_task != 10:
        base = {}                                        # CSV numbers are for the 10-per-task samples only
    if a.nocomp:
        for name, _crit, _k, drop, _note in depth:
            pruner.set(drop)
            base[("depth", name)] = f"{compare(run(), full)['arm_t0'] / scale:.3f}"
            pruner.set([])
        sc = np.load(a.width_scores.expanduser())
        for ratio in ratios:
            for l in range(n):
                s = sc[f"act_expert_{l}"]
                w = layers["expert"][l].mlp.down_proj.weight
                new = w.detach().double().cpu().clone()
                new[:, torch.from_numpy(np.argsort(s, kind="stable")[:int(round(ratio * s.size))].copy())] = 0
                patcher.set(w, new)
            base[("width", f"wexp_act{ratio}")] = f"{compare(run(), full)['arm_t0'] / scale:.3f}"
            patcher.restore()
        print("no-compensation reference: " + ", ".join(f"{k[1]} {v}" for k, v in base.items()), flush=True)

    PFX = {"lm": "model.vlm_with_expert.vlm.model.text_model.layers.",
           "expert": "model.vlm_with_expert.lm_expert.layers."}

    def save_ckpt(name, new_weights, info):
        """Unpruned 32-layer copy of the HF checkpoint with the compensated down_proj weights."""
        import json
        import shutil
        from safetensors import safe_open
        from safetensors.torch import save_file
        src = a.ckpt.expanduser()
        dst = a.save_dir.expanduser() / f"{name}_comp32"
        dst.mkdir(parents=True, exist_ok=True)
        f = safe_open(str(src / "model.safetensors"), framework="pt")
        missing = set(new_weights) - set(f.keys())
        assert not missing, f"keys not in checkpoint: {sorted(missing)[:3]}"
        tensors = {}
        for k in f.keys():
            t = f.get_tensor(k)
            if k in new_weights:
                assert tuple(new_weights[k].shape) == tuple(t.shape), k
                t = new_weights[k].to(t.dtype)
            tensors[k] = t.contiguous()
        save_file(tensors, str(dst / "model.safetensors"), metadata=f.metadata() or None)
        for p in src.iterdir():
            if p.is_file() and (p.name.startswith("policy_") or p.name == "config.json"):
                shutil.copy2(p, dst / p.name)
        (dst / "comp_info.json").write_text(json.dumps({"config": name, "source": str(src),
                                                        "fit_samples": len(fit_sel), **info}, indent=2) + "\n")
        print(f"   saved {dst} ({len(new_weights)} down_proj tensors compensated)", flush=True)

    rows_out = []

    def record(kind, name, alpha, note, t1):
        res = compare(run(), full)
        rows_out.append({"kind": kind, "name": name, "alpha": alpha, "note": note,
                         "rel_comp": f"{res['arm_t0'] / scale:.3f}", "rel_nocomp": base.get((kind, name), ""),
                         "arm_p90": f"{res['arm_p90']:.3e}", "arm_all": f"{res['arm_all']:.3e}",
                         "grip_t0": f"{res['grip_t0']:.3f}", "seconds": f"{time.time() - t1:.0f}"})
        r = rows_out[-1]
        print(f"{kind:5s} {name:12s} alpha {alpha:<6g} rel {r['rel_comp']} (no comp {r['rel_nocomp'] or '?'}) "
              f"p90 {r['arm_p90']} grip_t0 {r['grip_t0']} {note} {r['seconds']}s", flush=True)

    for alpha in alphas:
        for name, _crit, _k, drop, _note in depth:
            t1 = time.time()
            pruner.set(drop)
            done, new_w = [], {}
            for ra, rb in runs_of(drop):
                for mod in ("lm", "expert"):
                    if (mod, ra, rb) not in runs:
                        continue
                    w = layers[mod][ra - 1].mlp.down_proj.weight
                    M = ridge_solve(G64[(mod, ra - 1)], X64[(mod, ra, rb)], alpha)
                    new = w.detach().double().cpu() + M
                    patcher.set(w, new)
                    new_w[f"{PFX[mod]}{ra - 1}.mlp.down_proj.weight"] = new
                    done.append(f"{mod}:{ra}-{rb}@{ra - 1}")
            record("depth", name, alpha, " ".join(done), t1)
            if name in to_save:
                save_ckpt(name, new_w, {"kind": "depth", "drop": drop, "alpha": alpha, "compensated": done,
                                        "next": f"make_pruned_gguf.sh with SRC=<this dir> and drop "
                                                f"{','.join(map(str, drop))}"})
            patcher.restore()
            pruner.set([])
        for ratio in ratios:
            t1 = time.time()
            sc = np.load(a.width_scores.expanduser())
            new_w = {}
            for l in range(n):
                s = sc[f"act_expert_{l}"]
                drop_n = np.argsort(s, kind="stable")[:int(round(ratio * s.size))]
                K = torch.from_numpy(np.sort(np.setdiff1d(np.arange(s.size), drop_n)))
                w = layers["expert"][l].mlp.down_proj.weight
                C = G64[("expert", l)]
                W = w.detach().double().cpu()
                Wk = ridge_solve(C[K][:, K], W @ C[:, K], alpha)
                new = torch.zeros_like(W)
                new[:, K] = Wk
                patcher.set(w, new)
                new_w[f"{PFX['expert']}{l}.mlp.down_proj.weight"] = new
            record("width", f"wexp_act{ratio}", alpha, "", t1)
            if f"wexp_act{ratio}" in to_save:                 # dropped columns are 0; prune_width.py removes them
                save_ckpt(f"wexp_act{ratio}", new_w, {"kind": "width", "ratio": ratio, "alpha": alpha,
                                                       "next": f"make_width_gguf.sh with SRC=<this dir>, ratio {ratio}"})
            patcher.restore()

    again = run()
    print(f"restore check: max |full after all configs - full before| = {np.abs(again - full).max():.2e}")
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comp_dose.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"total {time.time() - t0:.0f}s, wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
