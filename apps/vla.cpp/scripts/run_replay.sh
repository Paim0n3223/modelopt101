#!/usr/bin/env bash
# Replay the calibration requests (replay_calib.py) on vla-server for several configs, one after another:
# for every config line start vla-server with that checkpoint/flags, wait for "ready", replay, stop it.
# Then compare the results with compare_replay.py against a reference replay (unpruned BF16, default flags).
#
# Usage (inside WSL, no other vla-server running):
#   bash run_replay.sh <out_dir> <configs_file>
#   configs_file lines: name|ENV1=a ENV2=b|--server-flag1 --server-flag2   (# comments, blank lines skipped)
#   VLA_GGUF=<path> in the env field serves another checkpoint for that config.
# Writes <out_dir>/server_<name>.npz and <out_dir>/server_<name>.log. A config whose .npz exists is skipped.
#
# Env overrides: VLA_DIR (~/vla.cpp), VLA_GGUF (BF16 smolvla-libero.gguf), CALIB (~/profile/calib),
#   PORT (5555), TPY (python with zmq + protobuf: ~/venvs/vla-tok/bin/python)
set -uo pipefail

OUT=${1:?usage: run_replay.sh <out_dir> <configs_file>}
CFG=${2:?usage: run_replay.sh <out_dir> <configs_file>}
HERE=$(cd "$(dirname "$0")" && pwd)
VLA=${VLA_DIR:-$HOME/vla.cpp}
GGUF_DEFAULT=${VLA_GGUF:-$HOME/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf}
CALIB=${CALIB:-$HOME/profile/calib}
PORT=${PORT:-5555}
TPY=${TPY:-$HOME/venvs/vla-tok/bin/python}
ADDR="tcp://127.0.0.1:$PORT"
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"

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

while IFS='|' read -r -u 3 name envs flags; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  npz="$OUT/server_$name.npz"; log="$OUT/server_$name.log"
  [[ -f "$npz" ]] && { echo "== $name: $npz exists, skipping"; continue; }
  # shellcheck disable=SC2086
  G=$(env $envs bash -c 'echo "${VLA_GGUF:-}"'); G=${G:-$GGUF_DEFAULT}
  [[ -f "$G" ]] || { echo "== $name: checkpoint not found: $G" >&2; continue; }
  echo "== $(date +%T) $name  gguf=$(basename "$G")  flags=[${flags}]"
  # shellcheck disable=SC2086
  env $envs "$VLA/build/vla-server" --bind "$ADDR" $flags "$G" > "$log" 2>&1 &
  SERVER_PID=$!
  ready=0
  for _ in $(seq 1 300); do
    grep -q "bound to .* ready" "$log" 2>/dev/null && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  if [[ $ready -eq 0 ]]; then echo "   vla-server did not become ready; see $log" >&2; stop_server; continue; fi
  "$TPY" "$HERE/replay_calib.py" --calib "$CALIB" --addr "$ADDR" --out "$npz" 2>&1 | grep -E "^vs recorded|^wrote"
  stop_server
done 3< "$CFG"
echo "== $(date +%T) done"
