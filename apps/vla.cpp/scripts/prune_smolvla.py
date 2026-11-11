"""Remove layer pairs (VLM text layer i + action-expert layer i) from a SmolVLA checkpoint.

SmolVLA's action-expert layer i reads the K/V of VLM layer i, so layers are removed as
"twin" pairs (as in CLP, arXiv 2606.20246). vla.cpp decides an expert layer's type from
its index (i % self_attn_every_n_layers == 0 -> self-attention, else cross-attention, and
the two types have differently shaped k/v weights), so after renumbering every kept layer
must keep its type. With self_attn_every_n_layers = 2 this means the kept layers must still
alternate SA, CA, SA, CA ... starting with SA. Removing adjacent pairs (i, i+1), or only the
last layers (nothing is renumbered), always works; e.g. removing 5 and 7 does not.

Usage (inside WSL, LIBERO venv: needs torch + safetensors):
  python prune_smolvla.py --src ~/models/smolvla_libero_hf --drop 30,31 --out ~/models/smolvla-prune/drop_30_31
  python prune_smolvla.py --src ... --drop 24-31 --out ...        (ranges allowed)
  python prune_smolvla.py --src ... --drop "" --out ...            (no-op copy, for checking)

Writes <out>/model.safetensors (layers renumbered), config.json (num_vlm_layers = kept
count, num_expert_layers unchanged), the policy pre/post-processor files, and
prune_info.json (kept/dropped original indices). Then convert with vla.cpp's
scripts/convert_smolvla_to_gguf.py --ckpt <out>.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

TXT = "model.vlm_with_expert.vlm.model.text_model.layers."
AEX = "model.vlm_with_expert.lm_expert.layers."
LAYER_RE = re.compile(r"^(" + re.escape(TXT) + "|" + re.escape(AEX) + r")(\d+)\.(.*)$")


def parse_drop(spec: str) -> list[int]:
    out = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(out)


def check_types(kept: list[int], every_n: int) -> list[str]:
    """Return problems if a kept layer would change between self- and cross-attention."""
    probs = []
    for new, old in enumerate(kept):
        sa_old = every_n > 0 and old % every_n == 0
        sa_new = every_n > 0 and new % every_n == 0
        if sa_old != sa_new:
            probs.append(f"layer {old} ({'SA' if sa_old else 'CA'}) would become index {new} "
                         f"({'SA' if sa_new else 'CA'})")
    return probs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="HF SmolVLA checkpoint dir")
    ap.add_argument("--drop", required=True, help='layer indices to remove, e.g. "30,31" or "24-31"; "" = none')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", default="", help="free-text tag stored in prune_info.json (e.g. cka, last, cosine)")
    a = ap.parse_args()

    src, out = a.src.expanduser().resolve(), a.out.expanduser().resolve()
    if out == src:
        raise SystemExit("--out must differ from --src")
    cfg = json.loads((src / "config.json").read_text())
    every_n = int(cfg["self_attn_every_n_layers"])

    f = safe_open(str(src / "model.safetensors"), framework="pt")
    keys = list(f.keys())
    idx = {int(m.group(2)) for k in keys if (m := LAYER_RE.match(k)) and m.group(1) == TXT}
    n_layers = max(idx) + 1
    if idx != set(range(n_layers)):
        raise SystemExit(f"text layers are not contiguous: {sorted(idx)}")
    idx_aex = {int(m.group(2)) for k in keys if (m := LAYER_RE.match(k)) and m.group(1) == AEX}
    if idx_aex != idx:
        raise SystemExit(f"checkpoint has {sorted(idx)} VLM layers but {sorted(idx_aex)} expert layers")
    if int(cfg.get("num_vlm_layers", 0)) not in (0, n_layers):
        raise SystemExit(f"config num_vlm_layers={cfg['num_vlm_layers']} but checkpoint has {n_layers} layers")
    if int(cfg.get("num_expert_layers", -1)) not in (-1, n_layers):
        raise SystemExit("num_expert_layers differs from the VLM layer count; not supported")

    drop = parse_drop(a.drop)
    bad = [d for d in drop if not 0 <= d < n_layers]
    if bad:
        raise SystemExit(f"--drop indices out of range [0, {n_layers}): {bad}")
    kept = [i for i in range(n_layers) if i not in drop]
    if not kept:
        raise SystemExit("cannot drop every layer")
    probs = check_types(kept, every_n)
    if probs:
        raise SystemExit("layer types would change after renumbering (vla.cpp derives SA/CA from the index):\n  "
                         + "\n  ".join(probs[:8]) + "\nDrop layers in adjacent pairs (i, i+1) instead.")
    new_of = {old: new for new, old in enumerate(kept)}

    tensors, n_dropped = {}, 0
    for k in keys:
        m = LAYER_RE.match(k)
        if m:
            old = int(m.group(2))
            if old not in new_of:
                n_dropped += 1
                continue
            k_new = f"{m.group(1)}{new_of[old]}.{m.group(3)}"
        else:
            k_new = k
        tensors[k_new] = f.get_tensor(k).contiguous()

    out.mkdir(parents=True, exist_ok=True)
    print(f"{n_layers} layers -> {len(kept)} (dropped {drop}); tensors {len(keys)} -> {len(tensors)} "
          f"({n_dropped} removed)", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata=f.metadata() or None)

    cfg_new = dict(cfg)
    cfg_new["num_vlm_layers"] = len(kept)          # vla.cpp converter and loader read this
    (out / "config.json").write_text(json.dumps(cfg_new, indent=4) + "\n")
    for p in src.iterdir():
        if p.is_file() and p.name.startswith("policy_"):
            shutil.copy2(p, out / p.name)
    info = {"source": str(src), "method": a.method, "n_layers_before": n_layers,
            "n_layers_after": len(kept), "dropped": drop, "kept": kept}
    (out / "prune_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
