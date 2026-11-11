#!/usr/bin/env bash
# LIBERO success-rate runs for SmolVLA on vla.cpp, one config after another.
#
# For every config line it starts vla-server with that checkpoint/flags, waits for "ready",
# runs eval/client/run_sim_client_direct.py on each task in $TASKS with N episodes,
# keeps the small result files (episodes.csv, summary.txt, logs) in <out_root>/<name>/
# and the videos under $VIDEO_ROOT/<name>/, then stops the server.
# A task whose summary.txt already exists in <out_root>/<name>/ is skipped, so an
# interrupted run can simply be restarted with the same command.
#
# Usage (inside WSL):
#   bash run_libero_sr.sh <out_root> <n_episodes> <configs_file>
#   configs_file lines: name|ENV1=a ENV2=b|--server-flag1 --server-flag2   (# comments, blank lines skipped)
#   VLA_GGUF=<path> in the env field serves another checkpoint for that config.
#
# Env overrides:
#   VLA_DIR     (~/vla.cpp)            VLA_GGUF  (BF16 smolvla-libero.gguf)
#   TASK_SUITE  (libero_object)        TASKS     ("0 1 2 3 4 5 6 7 8 9")
#   RENDER      (gpu = Mesa d3d12 on the NVIDIA adapter | cpu = llvmpipe)
#   SEED        (42, client env seed; episode k always uses init state k)
#   PORT        (5555)                 VIDEO_ROOT (~/vla.cpp/outputs/sr)
#   COOL        (30 s pause between configs)
set -uo pipefail

OUT_ROOT=${1:?usage: run_libero_sr.sh <out_root> <n_episodes> <configs_file>}
NEP=${2:?usage: run_libero_sr.sh <out_root> <n_episodes> <configs_file>}
CFG=${3:?usage: run_libero_sr.sh <out_root> <n_episodes> <configs_file>}
VLA=${VLA_DIR:-$HOME/vla.cpp}
GGUF_DEFAULT=${VLA_GGUF:-$HOME/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf}
SUITE=${TASK_SUITE:-libero_object}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
RENDER=${RENDER:-gpu}
SEED=${SEED:-42}
PORT=${PORT:-5555}
VIDEO_ROOT=${VIDEO_ROOT:-$VLA/outputs/sr}
COOL=${COOL:-30}
ADDR="tcp://127.0.0.1:$PORT"
PY="$VLA/eval/sim/libero/libero_uv/.venv/bin/python"
CLIENT="$VLA/eval/client/run_sim_client_direct.py"
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}

[[ "$NEP" =~ ^[1-9][0-9]*$ ]] || { echo "n_episodes must be a positive integer" >&2; exit 1; }
[[ -x "$VLA/build/vla-server" && -x "$PY" ]] || { echo "vla-server or LIBERO venv missing under $VLA" >&2; exit 1; }
mkdir -p "$OUT_ROOT"
OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
CFG=$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")

# Client-side render environment (see NOTES.md §8: the AMD adapter segfaults in MjrContext).
CLIENT_ENV=(MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false)
[[ "$RENDER" == gpu ]] && CLIENT_ENV+=(GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA)

SERVER_PID="" SMI_PID=""
stop_bg() {
  [[ -n "$SMI_PID" ]] && kill "$SMI_PID" 2>/dev/null; SMI_PID=""
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -KILL "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
}
trap 'stop_bg; exit 130' INT TERM
trap stop_bg EXIT

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already in use (another vla-server running?); stop it first" >&2; exit 1
fi

while IFS='|' read -r -u 3 name envs flags; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  d="$OUT_ROOT/$name"; vids="$VIDEO_ROOT/$name"
  mkdir -p "$d" "$vids"
  # shellcheck disable=SC2086
  G=$(env $envs bash -c 'echo "${VLA_GGUF:-}"'); G=${G:-$GGUF_DEFAULT}
  [[ -f "$G" ]] || { echo "== $name: checkpoint not found: $G" >&2; continue; }

  todo=()
  for t in $TASKS; do [[ -f "$d/task_$t/summary.txt" ]] || todo+=("$t"); done
  if [[ ${#todo[@]} -eq 0 ]]; then echo "== $name: all tasks already done, skipping"; continue; fi

  echo "== $(date +%T) $name  gguf=$(basename "$G")  flags=[${flags}]  tasks=[${todo[*]}]  episodes=$NEP"
  {
    echo "name: $name"
    echo "date: $(date '+%F %T %z')"
    echo "vla.cpp commit: $(git -C "$VLA" rev-parse --short HEAD 2>/dev/null)"
    echo "gguf: $G"
    echo "gguf size: $(stat -c %s "$G") bytes"
    echo "env: $envs"
    echo "server flags: $flags"
    echo "suite: $SUITE  tasks: $TASKS  episodes/task: $NEP  seed: $SEED  n_action_steps: 1"
    echo "render: $RENDER (${CLIENT_ENV[*]})"
    nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw --format=csv
  } >> "$d/env.txt"

  # GPU samples every 2 s for the whole config (temperature/clock throttle record).
  nvidia-smi --query-gpu=timestamp,memory.used,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active \
    --format=csv -l 2 >> "$d/gpu_samples.csv" 2>/dev/null &
  SMI_PID=$!

  # shellcheck disable=SC2086
  env $envs "$VLA/build/vla-server" --bind "$ADDR" $flags "$G" >> "$d/server.log" 2>&1 &
  SERVER_PID=$!
  ready=0
  for _ in $(seq 1 300); do
    grep -q "bound to .* ready" "$d/server.log" 2>/dev/null && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  if [[ $ready -eq 0 ]]; then
    echo "   vla-server did not become ready; see $d/server.log" >&2
    tail -n 20 "$d/server.log" >&2
    stop_bg; continue
  fi

  for t in "${todo[@]}"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "   vla-server died; stopping $name" >&2; break; fi
    rm -rf "$vids/smolvla/$SUITE/task_$t"     # a half-finished earlier attempt
    t0=$(date +%s)
    env "${CLIENT_ENV[@]}" "$PY" "$CLIENT" --arch smolvla --vla-addr "$ADDR" \
      --task "$SUITE" --task-id "$t" --n-episodes "$NEP" --n-action-steps 1 --seed "$SEED" \
      --output-dir "$vids" > "$d/client_task_$t.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    src="$vids/smolvla/$SUITE/task_$t"
    if [[ $rc -eq 0 && -f "$src/summary.txt" ]]; then
      mkdir -p "$d/task_$t"
      cp "$src/episodes.csv" "$d/task_$t/"
      { cat "$src/summary.txt"; echo "Wall time (s): $dt"; } > "$d/task_$t/summary.txt"
      echo "   $(date +%T) task $t: $(grep -o 'Success rate:.*' "$src/summary.txt")  (${dt}s)"
    else
      echo "   $(date +%T) task $t FAILED (exit $rc); see $d/client_task_$t.log" >&2
    fi
  done

  stop_bg
  sleep "$COOL"
done 3< "$CFG"
echo "== $(date +%T) done"
