"""Compare vla-server replays (replay_calib.py) against a reference replay, with the same
metrics as prune_diag.py, so configs with a known LIBERO success rate (quantization,
fewer denoising steps, ...) can be placed on the same scale as the pruned models.

  arm_t0   mean |delta| of the 6 arm dims at chunk step 0 (the step the client executes)
  rel      arm_t0 / mean |reference arm action at step 0|
  arm_p90  90th percentile over samples of the per-sample arm_t0
  arm_all  mean |delta| of the arm dims over all 50 chunk steps
  grip_t0  fraction of samples whose gripper command changes sign at step 0

Usage (any venv with numpy, e.g. ~/venvs/vla-tok):
  python compare_replay.py --ref results/prune/diag/server_full.npz \
      q4v_fa_mmdef=results/prune/diag/server_q4v_fa_mmdef.npz  [name=npz ...]
"""
import argparse

import numpy as np

ARM, GRIP = slice(0, 6), 6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference replay npz (unpruned BF16, default flags)")
    ap.add_argument("items", nargs="+", metavar="NAME=NPZ")
    a = ap.parse_args()

    ref = np.load(a.ref)
    r = ref["action"][..., :7]
    scale = np.abs(r[:, 0, ARM]).mean()
    print(f"reference {a.ref}: {len(r)} samples, mean |arm action| at t0 {scale:.3e}")
    print(f"{'name':22s} {'arm_t0':>10s} {'rel':>6s} {'arm_p90':>10s} {'arm_all':>10s} {'grip_t0':>7s}")
    for item in a.items:
        name, _, path = item.partition("=")
        z = np.load(path)
        if not (np.array_equal(z["task"], ref["task"]) and np.array_equal(z["row"], ref["row"])):
            raise SystemExit(f"{path}: samples differ from the reference (use the same --per-task)")
        p = z["action"][..., :7]
        arm0 = np.abs(p[:, 0, ARM] - r[:, 0, ARM]).mean(1)
        print(f"{name:22s} {arm0.mean():10.3e} {arm0.mean() / scale:6.3f} {np.percentile(arm0, 90):10.3e} "
              f"{np.abs(p[..., ARM] - r[..., ARM]).mean():10.3e} "
              f"{np.mean(np.sign(p[:, 0, GRIP]) != np.sign(r[:, 0, GRIP])):7.3f}")


if __name__ == "__main__":
    main()
