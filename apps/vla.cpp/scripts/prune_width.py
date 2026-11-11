"""Width-prune the action expert's FFN of a SmolVLA checkpoint (remove neurons for real).

Neuron j of expert layer l = row j of mlp.gate_proj / mlp.up_proj and column j of
mlp.down_proj. Every expert layer keeps the same number of neurons (vla.cpp has one
expert FFN size, smolvla.expert_inter, which its converter reads from the shape of
layer 0's gate_proj). Which neurons go is read from width_diag.py's scores.npz: the
lowest-scoring round(ratio * 1280) of each layer, the same set width_diag.py zeroed.

Only the expert is supported: vla.cpp's converter hard-codes the LM FFN size (2560,
SMOLLM2_500M in convert_smolvla_to_gguf.py), and LM width pruning breaks the model anyway
(width_diag.py). config.json is copied unchanged (lerobot would derive another expert
FFN size from it, so the output is for vla.cpp only).

Usage (inside WSL, LIBERO venv: needs torch + safetensors):
  python prune_width.py --src ~/models/smolvla_libero_hf --scores results/prune/width/scores.npz \
      --criterion act --ratio 0.25 --out ~/models/smolvla-prune/wexp_act25
Then convert with vla.cpp's scripts/convert_smolvla_to_gguf.py --ckpt <out> (make_width_gguf.sh).
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

AEX = "model.vlm_with_expert.lm_expert.layers."
FFN_RE = re.compile(r"^" + re.escape(AEX) + r"(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="HF SmolVLA checkpoint dir")
    ap.add_argument("--scores", type=Path, required=True, help="scores.npz from width_diag.py")
    ap.add_argument("--criterion", default="act", choices=["act", "mag", "rand"])
    ap.add_argument("--ratio", type=float, required=True, help="fraction of expert FFN neurons removed per layer")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    src, out = a.src.expanduser().resolve(), a.out.expanduser().resolve()
    if out == src:
        raise SystemExit("--out must differ from --src")
    if not 0 < a.ratio < 1:
        raise SystemExit("--ratio must be in (0, 1)")
    sc = np.load(a.scores.expanduser())

    f = safe_open(str(src / "model.safetensors"), framework="pt")
    keys = list(f.keys())
    layers = sorted({int(m.group(1)) for k in keys if (m := FFN_RE.match(k))})
    if not layers:
        raise SystemExit("no expert FFN tensors found")

    keep = {}
    for l in layers:
        s = sc[f"{a.criterion}_expert_{l}"]
        n_drop = int(round(a.ratio * s.size))
        drop = np.argsort(s, kind="stable")[:n_drop]               # same rule as width_diag.py
        keep[l] = torch.from_numpy(np.sort(np.setdiff1d(np.arange(s.size), drop)))
    n_keep = {len(v) for v in keep.values()}
    assert len(n_keep) == 1, "every layer must keep the same number of neurons"
    n_keep = n_keep.pop()

    tensors, n_changed = {}, 0
    for k in keys:
        t = f.get_tensor(k)
        m = FFN_RE.match(k)
        if m:
            l, proj, kind = int(m.group(1)), m.group(2), m.group(3)
            idx = keep[l]
            if proj == "down_proj":
                t = t[:, idx] if kind == "weight" else t                 # down bias is per output: unchanged
            else:
                t = t[idx]                                            # gate/up: rows (and bias entries)
            n_changed += 1
        tensors[k] = t.contiguous()

    out.mkdir(parents=True, exist_ok=True)
    first = f"{AEX}{layers[0]}.mlp.gate_proj.weight"
    print(f"expert FFN {f.get_slice(first).get_shape()[0]} -> {n_keep} neurons in {len(layers)} layers "
          f"({a.criterion}, ratio {a.ratio}); {n_changed} tensors sliced", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata=f.metadata() or None)
    for p in src.iterdir():
        if p.is_file() and (p.name.startswith("policy_") or p.name == "config.json"):
            shutil.copy2(p, out / p.name)
    info = {"source": str(src), "pruning": "expert FFN width", "criterion": a.criterion, "ratio": a.ratio,
            "scores": str(a.scores), "expert_inter_after": n_keep,
            "kept": {str(l): v.tolist() for l, v in keep.items()}}
    (out / "prune_info.json").write_text(json.dumps(info) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
