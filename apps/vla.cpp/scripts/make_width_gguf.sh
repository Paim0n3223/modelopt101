#!/usr/bin/env bash
# Build an expert-FFN width-pruned SmolVLA GGUF: prune_width.py (safetensors) -> vla.cpp converter (GGUF),
# optionally also a Q8_0 copy (LM + vision quantized, expert stays BF16 as in §7).
# Nothing is written inside the vla.cpp clone (no __pycache__, output under $PRUNE_ROOT).
#
# Usage (inside WSL):
#   bash make_width_gguf.sh <name> <ratio> [criterion=act] [q8]
#   e.g. bash make_width_gguf.sh wexp_act25 0.25 act q8
# Result: $PRUNE_ROOT/<name>/ , $PRUNE_ROOT/<name>.gguf and, with "q8", $PRUNE_ROOT/<name>-q8_0-vis.gguf
#
# Env overrides: SRC (~/models/smolvla_libero_hf), PRUNE_ROOT (~/models/smolvla-prune),
#   SCORES (results/prune/width/scores.npz of this repo), VLA_DIR (~/vla.cpp),
#   PY (LIBERO venv python: torch + safetensors), QPY (~/venvs/vla-tok python, for quantize)
set -euo pipefail

NAME=${1:?usage: make_width_gguf.sh <name> <ratio> [criterion] [q8]}
RATIO=${2:?usage: make_width_gguf.sh <name> <ratio> [criterion] [q8]}
CRIT=${3:-act}
Q8=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/models/smolvla_libero_hf}
ROOT=${PRUNE_ROOT:-$HOME/models/smolvla-prune}
SCORES=${SCORES:-$HERE/../results/prune/width/scores.npz}
VLA=${VLA_DIR:-$HOME/vla.cpp}
PY=${PY:-$VLA/eval/sim/libero/libero_uv/.venv/bin/python}
QPY=${QPY:-$HOME/venvs/vla-tok/bin/python}
GGUFPY="$VLA/build/_deps/llama-src/gguf-py:$VLA/scripts"
export PYTHONDONTWRITEBYTECODE=1

"$PY" "$HERE/prune_width.py" --src "$SRC" --scores "$SCORES" --criterion "$CRIT" --ratio "$RATIO" --out "$ROOT/$NAME"
PYTHONPATH="$GGUFPY" "$PY" "$VLA/scripts/convert_smolvla_to_gguf.py" --ckpt "$ROOT/$NAME" --out "$ROOT/$NAME.gguf" \
  | grep -E "resolved cfg|done\."
if [[ "$Q8" == q8 ]]; then
  PYTHONPATH="$GGUFPY" "$QPY" "$VLA/scripts/quantize_gguf.py" --in "$ROOT/$NAME.gguf" \
    --out "$ROOT/$NAME-q8_0-vis.gguf" --type Q8_0 --vision | tail -2
fi
