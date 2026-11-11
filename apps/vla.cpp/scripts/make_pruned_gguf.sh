#!/usr/bin/env bash
# Build a layer-pruned SmolVLA GGUF: prune_smolvla.py (safetensors) -> vla.cpp converter (GGUF).
# Nothing is written inside the vla.cpp clone (no __pycache__, output under $PRUNE_ROOT).
#
# Usage (inside WSL):
#   bash make_pruned_gguf.sh <name> "<drop spec>" [method]
#   e.g. bash make_pruned_gguf.sh drop_30_31 "30,31" last
#        bash make_pruned_gguf.sh noop "" none          (no layer removed; must equal full32.gguf)
# Result: $PRUNE_ROOT/<name>/ (safetensors + config + prune_info.json) and $PRUNE_ROOT/<name>.gguf
#
# Env overrides: SRC (~/models/smolvla_libero_hf), PRUNE_ROOT (~/models/smolvla-prune),
#   VLA_DIR (~/vla.cpp), PY (LIBERO venv python: has torch + safetensors)
set -euo pipefail

NAME=${1:?usage: make_pruned_gguf.sh <name> "<drop spec>" [method]}
DROP=${2-}
METHOD=${3:-}
SRC=${SRC:-$HOME/models/smolvla_libero_hf}
ROOT=${PRUNE_ROOT:-$HOME/models/smolvla-prune}
VLA=${VLA_DIR:-$HOME/vla.cpp}
PY=${PY:-$VLA/eval/sim/libero/libero_uv/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1

"$PY" "$HERE/prune_smolvla.py" --src "$SRC" --drop "$DROP" --method "$METHOD" --out "$ROOT/$NAME"
PYTHONPATH="$VLA/build/_deps/llama-src/gguf-py:$VLA/scripts" \
  "$PY" "$VLA/scripts/convert_smolvla_to_gguf.py" --ckpt "$ROOT/$NAME" --out "$ROOT/$NAME.gguf" \
  | grep -E "resolved cfg|done\."
