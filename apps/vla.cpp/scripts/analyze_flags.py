"""Summarise a run_flags.sh output directory as a markdown table (standard library only).

Usage: python3 analyze_flags.py <out_dir> [reference_config_name=default]

Per config: latency from vla-bench, peak whole-GPU memory.used from gpu_samples.csv, GPU
temperature at the start, and the deviation of vla_predict_check's action chunk against the
reference config (first 7 of 32 columns = the real LIBERO action dims, 50 rows).
"""
import csv
import re
import sys
from pathlib import Path

REAL_DIMS, MAX_DIM = 7, 32
BENCH_RE = re.compile(r"min ([\d.]+) ms\s+mean ([\d.]+) ms\s+p50 ([\d.]+) ms\s+p90 ([\d.]+) ms"
                      r"(?:\s+vision ([\d.]+) ms)?")


def read_actions(path):
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("action_len="))
    except StopIteration:
        return None
    vals = [float(x) for x in lines[start + 1:] if x.strip()]
    rows = [vals[r * MAX_DIM:r * MAX_DIM + REAL_DIMS] for r in range(len(vals) // MAX_DIM)]
    return rows or None


def bench(d):
    p = d / "bench_p1.txt"
    if not p.exists():
        return None, "no bench output"
    text = p.read_text()
    m = BENCH_RE.search(text)
    if m:
        return [float(x) if x else None for x in m.groups()], ""
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith(("ggml_", "  Device", "#"))]
    err = [l for l in lines if "only supported" in l or "error" in l.lower() or "failed" in l]
    return None, (err[0] if err else (lines[-1] if lines else "no summary line"))[:90]


def peak_mem(d):
    p = d / "gpu_samples.csv"
    if not p.exists():
        return None
    peak = 0
    with p.open() as f:
        for row in csv.reader(f):
            if len(row) > 1 and "MiB" in row[1] and "memory" not in row[1]:
                peak = max(peak, int(row[1].split()[0]))
    return peak


def start_temp(d):
    p = d / "env.txt"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if "Laptop GPU" in line or line.count(",") >= 6 and "MiB" in line:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4 and parts[3].isdigit():
                return int(parts[3])
    return None


def switch_rows(rows):
    """Rows where the gripper (dim 6) changes sign, i.e. open/close commands."""
    return [i for i in range(1, len(rows)) if (rows[i][6] > 0) != (rows[i - 1][6] > 0)]


def deviation(ref, got):
    """(max, mean |diff| over 7 dims, mean |ref|, gripper sign flips,
    arm-only max, arm-only mean over dims 0-5, gripper switch rows ref -> got)."""
    if ref is None or got is None or len(ref) != len(got):
        return None
    diffs = [abs(a - b) for ra, rb in zip(ref, got) for a, b in zip(ra, rb)]
    arm = [abs(a - b) for ra, rb in zip(ref, got) for a, b in zip(ra[:6], rb[:6])]
    flips = sum(1 for ra, rb in zip(ref, got) if (ra[6] > 0) != (rb[6] > 0))
    mean_abs_ref = sum(abs(a) for r in ref for a in r) / len(diffs)
    sw = f"{switch_rows(ref)}->{switch_rows(got)}"
    return max(diffs), sum(diffs) / len(diffs), mean_abs_ref, flips, max(arm), sum(arm) / len(arm), sw


def main():
    out = Path(sys.argv[1])
    ref_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    ref = read_actions(out / ref_name / "pred.txt")
    base = bench(out / ref_name)[0]
    dirs = [d for d in sorted(out.iterdir(), key=lambda p: p.stat().st_mtime) if d.is_dir()]

    print("| Config | Flags / env | min ms | mean ms | p50 ms | p90 ms | vision ms | mean vs default "
          "| peak VRAM MiB | T start °C | max abs dev | mean abs dev | gripper sign flips "
          "| arm max dev | arm mean dev | gripper switch rows |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for d in dirs:
        cfg = (d / "config.txt").read_text().strip() if (d / "config.txt").exists() else ""
        envs, _, flags = cfg.partition("|")
        label = " ".join(x for x in (envs.strip(), flags.strip()) if x) or "(defaults)"
        b, err = bench(d)
        dev = deviation(ref, read_actions(d / "pred.txt"))
        mem, temp = peak_mem(d), start_temp(d)
        if b:
            rel = f"{(b[1] / base[1] - 1) * 100:+.1f}%" if base else "-"
            lat = [f"{x:.1f}" if x is not None else "-" for x in b] + [rel]
        else:
            lat = [f"failed: {err}", "", "", "", "", ""]
        devs = ([f"{dev[0]:.2e}", f"{dev[1]:.2e}", f"{dev[3]}/50", f"{dev[4]:.2e}", f"{dev[5]:.2e}", dev[6]]
                if dev else ["-"] * 6)
        print(f"| {d.name} | `{label}` | " + " | ".join(lat) + f" | {mem if mem is not None else '-'} | "
              f"{temp if temp is not None else '-'} | " + " | ".join(devs) + " |")
    if ref:
        mean_abs_ref = sum(abs(a) for r in ref for a in r) / (len(ref) * REAL_DIMS)
        print(f"\nReference: `{ref_name}`, mean |action| over the 7 real dims = {mean_abs_ref:.3e}")


if __name__ == "__main__":
    main()
