#!/usr/bin/env bash
# Per-phase latency of SmolVLA on vla-server, for each config in a configs file and
# each timing mode (none = real latency, phase = prefill/denoise split, slower).
#
# For every (config, mode) it starts vla-server --timing-detail <mode>, waits for
# "ready", runs scripts/phase_client.py (warmup + N timed requests), then stops the
# server and cools down. Nothing is written inside the vla.cpp clone.
#
# Usage (inside WSL):
#   bash run_phase.sh <out_dir> <configs_file>
#   configs_file lines: name|ENV1=a|--server-flag ...   (same format as run_libero_sr.sh)
#
# Env overrides: VLA_DIR (~/vla.cpp), VLA_GGUF (BF16 smolvla-libero.gguf),
#   MODES ("none phase"), N (50), WARMUP (3), VIEWS (2), PORT (5555), COOL (30),
#   PY (~/venvs/vla-tok/bin/python)
set -uo pipefail

OUT=${1:?usage: run_phase.sh <out_dir> <configs_file>}
CFG=${2:?usage: run_phase.sh <out_dir> <configs_file>}
VLA=${VLA_DIR:-$HOME/vla.cpp}
GGUF_DEFAULT=${VLA_GGUF:-$HOME/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf}
MODES=${MODES:-"none phase"}
N=${N:-50}; WARMUP=${WARMUP:-3}; VIEWS=${VIEWS:-2}
PORT=${PORT:-5555}; COOL=${COOL:-30}
PY=${PY:-$HOME/venvs/vla-tok/bin/python}
ADDR="tcp://127.0.0.1:$PORT"
HERE=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

[[ -x "$VLA/build/vla-server" && -x "$PY" ]] || { echo "vla-server or $PY missing" >&2; exit 1; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
CFG=$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -KILL "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
}
trap 'stop_server; exit 130' INT TERM
trap stop_server EXIT

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already in use (another vla-server running?); stop it first" >&2; exit 1
fi

{
  echo "date: $(date '+%F %T %z')"
  echo "vla.cpp commit: $(git -C "$VLA" rev-parse --short HEAD 2>/dev/null)"
  echo "client: phase_client.py --views $VIEWS --warmup $WARMUP --n $N"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} > "$OUT/env.txt"

while IFS='|' read -r -u 3 name envs flags; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  # shellcheck disable=SC2086
  G=$(env $envs bash -c 'echo "${VLA_GGUF:-}"'); G=${G:-$GGUF_DEFAULT}
  [[ -f "$G" ]] || { echo "== $name: checkpoint not found: $G" >&2; continue; }
  for mode in $MODES; do
    tag="${name}_${mode}"
    log="$OUT/$tag.server.log"
    : > "$log"
    echo "== $(date +%T) $tag  gguf=$(basename "$G")  flags=[${flags}]  start temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C"
    echo "$tag: gguf=$G flags=[$flags] env=[$envs]" >> "$OUT/env.txt"
    # shellcheck disable=SC2086
    env $envs "$VLA/build/vla-server" --bind "$ADDR" --timing-detail "$mode" $flags "$G" >> "$log" 2>&1 &
    SERVER_PID=$!
    ready=0
    for _ in $(seq 1 300); do
      grep -q "bound to .* ready" "$log" 2>/dev/null && { ready=1; break; }
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if [[ $ready -eq 0 ]]; then
      echo "   vla-server did not become ready; see $log" >&2; tail -n 20 "$log" >&2
      stop_server; continue
    fi
    "$PY" "$HERE/phase_client.py" --addr "$ADDR" --views "$VIEWS" --warmup "$WARMUP" --n "$N" \
      --proto "$VLA/src/serving/vla.proto" --label "$tag" --out "$OUT/$tag.csv" 2>&1 \
      | tee "$OUT/$tag.client.log" | sed 's/^/   /'
    stop_server
    sleep "$COOL"
  done
done 3< "$CFG"
echo "== $(date +%T) done"
