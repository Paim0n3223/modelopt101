#!/usr/bin/env bash
# Latency and memory of SmolVLA with vla-bench, following vla.cpp docs/benchmark/TEMPLATE.md:
#   3 warmups + 20 timed reps per process, 3 processes, 2 views at 512 px, 16 lang tokens.
#   GPU memory, temperature and clocks are sampled every 0.5 s with nvidia-smi.
#
# Usage (inside WSL):
#   bash run_baseline.sh <out_dir> [extra vla-bench flags, e.g. --flash-attn --mm-prec default]
# Env overrides: VLA_DIR (default ~/vla.cpp), VLA_GGUF, NPROC (default 3), WARMUP (3), REPS (20).
set -euo pipefail

OUT=${1:?usage: run_baseline.sh <out_dir> [extra vla-bench flags]}
shift
EXTRA=("$@")
VLA=${VLA_DIR:-$HOME/vla.cpp}
GGUF=${VLA_GGUF:-$HOME/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf}
NPROC=${NPROC:-3}
WARMUP=${WARMUP:-3}
REPS=${REPS:-20}
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
mkdir -p "$OUT"

{
  echo "date_start: $(date '+%F %T %z')"
  echo "vla.cpp: $(git -C "$VLA" log -1 --format='%h %ad' --date=short)"
  echo "gguf: $GGUF"
  echo "gguf_sha256: $(sha256sum "$GGUF" | cut -d' ' -f1)"
  echo "bench: --images 2 --size 512 --warmup $WARMUP --reps $REPS, $NPROC processes"
  echo "extra_flags: ${EXTRA[*]:-(none, defaults)}"
  echo "kernel: $(uname -r)"
  echo "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo n/a)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,clocks.sm,clocks.max.sm,pstate --format=csv
} > "$OUT/env.txt"

# Background samplers. compute_apps is the per-process probe TEMPLATE.md asks for;
# gpu_samples is the whole-GPU fallback (memory.used includes other processes).
nvidia-smi --query-gpu=timestamp,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -lms 500 > "$OUT/gpu_samples.csv" 2>&1 &
S1=$!
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory \
  --format=csv -lms 500 > "$OUT/compute_apps.csv" 2>&1 &
S2=$!
trap 'kill $S1 $S2 2>/dev/null || true' EXIT
sleep 3   # idle reference samples before the first process

for i in $(seq 1 "$NPROC"); do
  echo "proc_${i}_start: $(date +%T.%3N)" >> "$OUT/env.txt"
  # Python wrapper only to read the child's peak RSS (getrusage), as the protocol asks.
  python3 - "$OUT/bench_p$i.txt" "$VLA/build/vla-bench" --ckpt "$GGUF" --images 2 --size 512 \
      --warmup "$WARMUP" --reps "$REPS" "${EXTRA[@]}" <<'PY'
import resource, subprocess, sys
out = sys.argv[1]
with open(out, "w") as f:
    rc = subprocess.run(sys.argv[2:], stdout=f, stderr=subprocess.STDOUT).returncode
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    f.write(f"\n# exit={rc} peak_host_rss_MiB={rss:.1f}\n")
sys.exit(rc)
PY
  echo "proc_${i}_end: $(date +%T.%3N)" >> "$OUT/env.txt"
  sleep 3
done

echo "date_end: $(date '+%F %T %z')" >> "$OUT/env.txt"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,pstate --format=csv >> "$OUT/env.txt"
