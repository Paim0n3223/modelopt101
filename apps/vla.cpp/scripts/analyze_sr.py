#!/usr/bin/env python3
"""Summarise LIBERO success-rate runs written by run_libero_sr.sh.

Usage: python3 analyze_sr.py <out_root> [ref_config]
Prints a markdown table (one row per config dir in <out_root>) and a per-task table.
The 95% interval is the Wilson score interval: with 40 episodes it is roughly +-15
points, so screening differences smaller than that are not evidence of a real change.
If ref_config is given, the per-task table marks tasks where a config lost successes.
"""
import csv
import math
import re
import sys
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def load(cfg_dir):
    tasks = {}
    for tdir in sorted(cfg_dir.glob("task_*"), key=lambda p: int(p.name.split("_")[1])):
        ep_csv = tdir / "episodes.csv"
        if not ep_csv.is_file():
            continue
        rows = list(csv.DictReader(ep_csv.open()))
        summ = (tdir / "summary.txt").read_text() if (tdir / "summary.txt").is_file() else ""
        m = re.search(r"Average inference time per step:\s*([\d.]+)", summ)
        w = re.search(r"Wall time \(s\):\s*(\d+)", summ)
        tasks[int(tdir.name.split("_")[1])] = {
            "succ": [int(r["success"]) for r in rows],
            "steps": [int(r["steps"]) for r in rows],
            "aborted": sum(int(r.get("aborted", 0) or 0) for r in rows),
            "infer_ms": float(m.group(1)) if m else float("nan"),
            "wall_s": int(w.group(1)) if w else 0,
        }
    meta = {}
    # vla-server logs every 10th request: "rid=N served=M total=X ms vision=Y inf=Z other=W".
    # rid=0 pays the CUDA-graph warmup and is dropped; the median is over the rest.
    srv = cfg_dir / "server.log"
    if srv.is_file():
        pat = re.compile(r"rid=(\d+)\s+served=\d+\s+total=([\d.]+) ms\s+vision=([\d.]+)\s+inf=([\d.]+)")
        rows = [tuple(map(float, m.groups()[1:])) for m in pat.finditer(srv.read_text()) if m.group(1) != "0"]
        if rows:
            med = lambda xs: sorted(xs)[len(xs) // 2]
            meta["srv"] = tuple(med([r[i] for r in rows]) for i in range(3))
    env = cfg_dir / "env.txt"
    if env.is_file():
        for line in env.read_text().splitlines():
            if ":" in line and line.split(":", 1)[0] in ("gguf", "server flags"):
                k, v = line.split(":", 1)
                meta[k] = v.strip()
    return tasks, meta


def main():
    root = Path(sys.argv[1])
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    configs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        tasks, meta = load(d)
        if tasks:
            configs[d.name] = (tasks, meta)
    if not configs:
        print("no results under", root)
        return

    print("| Config | Checkpoint | Server flags | Tasks | Episodes | SR | 95% CI | Mean steps (success) "
          "| Client ms/step | Server total / vision / inf ms (median) | Wall min |")
    print("|---|---|---|--:|--:|--:|---|--:|--:|---|--:|")
    for name, (tasks, meta) in configs.items():
        succ = [s for t in tasks.values() for s in t["succ"]]
        steps_ok = [st for t in tasks.values() for s, st in zip(t["succ"], t["steps"]) if s]
        n, k = len(succ), sum(succ)
        lo, hi = wilson(k, n)
        n_ep = [len(t["succ"]) for t in tasks.values()]
        infer = sum(t["infer_ms"] * len(t["succ"]) for t in tasks.values()) / max(n, 1)
        wall = sum(t["wall_s"] for t in tasks.values()) / 60
        aborted = sum(t["aborted"] for t in tasks.values())
        ck = Path(meta.get("gguf", "?")).name
        flags = meta.get("server flags", "") or "(default)"
        sr = f"{100 * k / n:.1f}% ({k}/{n})" + (f", {aborted} aborted" if aborted else "")
        mean_steps = f"{sum(steps_ok) / len(steps_ok):.0f}" if steps_ok else "-"
        srv = meta.get("srv")
        srv_txt = f"{srv[0]:.1f} / {srv[1]:.1f} / {srv[2]:.1f}" if srv else "-"
        print(f"| {name} | {ck} | {flags} | {len(tasks)} | {n} | {sr} | {100 * lo:.0f}–{100 * hi:.0f}% "
              f"| {mean_steps} | {infer:.1f} | {srv_txt} | {wall:.0f} |")
        if len(set(n_ep)) > 1:
            print(f"<!-- warning: {name} has unequal episodes per task: {n_ep} -->")

    all_tasks = sorted({t for tasks, _ in configs.values() for t in tasks})
    print()
    print("Successes per task (k/n):")
    print()
    print("| Config | " + " | ".join(f"t{t}" for t in all_tasks) + " |")
    print("|---|" + "--:|" * len(all_tasks))
    ref_tasks = configs[ref][0] if ref in configs else None
    for name, (tasks, _) in configs.items():
        cells = []
        for t in all_tasks:
            if t not in tasks:
                cells.append("-")
                continue
            k, n = sum(tasks[t]["succ"]), len(tasks[t]["succ"])
            cell = f"{k}/{n}"
            if ref_tasks and name != ref and t in ref_tasks and k < sum(ref_tasks[t]["succ"]):
                cell = f"**{cell}**"
            cells.append(cell)
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
