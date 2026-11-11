#!/usr/bin/env bash
# Runtime-flag screen for SmolVLA. For every config line:
#   1. a short vla-bench run through run_baseline.sh (TEMPLATE.md screen: 2 warmups + 5 reps,
#      1 process, GPU sampled every 0.5 s),
#   2. vla_predict_check with the same flags (fixed synthetic images, tokens, state and noise),
#      so analyze_flags.py can measure the output deviation against the "default" config,
#   3. a cool-down pause so every config starts from a similar GPU temperature.
#
# Usage (inside WSL): bash run_flags.sh <out_dir> <configs_file>
#   configs_file lines: name|ENV1=a ENV2=b|--flag1 --flag2      (blank lines and # comments skipped)
#   VLA_GGUF=<path> in the env field runs that config on another checkpoint (e.g. a quantized one).
# Env overrides: VLA_DIR, VLA_GGUF, WARMUP (2), REPS (5), COOL (60 s), IMAGES_PC (1, views for predict_check)
set -uo pipefail

OUT=${1:?usage: run_flags.sh <out_dir> <configs_file>}
CFG=${2:?usage: run_flags.sh <out_dir> <configs_file>}
HERE=$(cd "$(dirname "$0")" && pwd)
VLA=${VLA_DIR:-$HOME/vla.cpp}
GGUF=${VLA_GGUF:-$HOME/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf}
COOL=${COOL:-60}
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda-13.1/bin:$PATH
mkdir -p "$OUT"

while IFS='|' read -r -u 3 name envs flags; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  d="$OUT/$name"
  mkdir -p "$d"
  echo "== $(date +%T) $name  env=[${envs}]  flags=[${flags}]"
  # shellcheck disable=SC2086
  env $envs NPROC=1 WARMUP="${WARMUP:-2}" REPS="${REPS:-5}" \
      bash "$HERE/run_baseline.sh" "$d" $flags || echo "   vla-bench failed for $name"
  # A config may pick its own checkpoint with VLA_GGUF=... in the env field.
  # shellcheck disable=SC2086
  G=$(env $envs bash -c 'echo "${VLA_GGUF:-}"'); G=${G:-$GGUF}
  # shellcheck disable=SC2086
  env $envs VLA_IMG_SIZE=512 "$VLA/build/tests/vla_predict_check" "$G" "" "${IMAGES_PC:-1}" $flags \
      > "$d/pred.txt" 2> "$d/pred.err" || echo "   vla_predict_check failed for $name"
  echo "$envs|$flags" > "$d/config.txt"
  sleep "$COOL"
done 3< "$CFG"
echo "== $(date +%T) done"
