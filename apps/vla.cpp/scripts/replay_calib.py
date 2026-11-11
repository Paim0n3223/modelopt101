"""Replay recorded calibration requests against a running vla-server and save its actions.

Used to check a pruned GGUF against the same pruning done in PyTorch (prune_diag.py
--server): if the vla.cpp actions match the PyTorch ones about as closely as the
unpruned model does (~7e-4 mean abs, results/prune/cka/sanity.txt), the
prune -> renumber -> convert -> load pipeline is correct.

The requests are rebuilt from collect_calib.py's npz files (images stored as uint8, so
they differ from the original float images by rounding; the recorded noise is reused).
Replaying the UNPRUNED GGUF first therefore shows how much that rounding alone moves
the action ("vs recorded" line below).

Usage (inside WSL, ~/venvs/vla-tok, vla-server with the GGUF to test already listening):
  python replay_calib.py --calib ~/profile/calib --out results/prune/diag/server_cka_k4.npz

Writes task, row, action [N, 50, action_dim] (as returned by the server).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def select_rows(files, per_task):
    """Evenly spaced rows of every task file -> list of (task_index, row).
    Shared with prune_diag.py so both scripts look at the same samples."""
    out = []
    for t, f in enumerate(files):
        n = len(np.load(f)["n_tokens"])
        rows = np.unique(np.linspace(0, n - 1, min(per_task, n)).round().astype(int))
        out += [(t, int(r)) for r in rows]
    return out


def calib_files(calib):
    files = sorted(Path(calib).expanduser().glob("task_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no task_*.npz in {calib}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--addr", default="tcp://127.0.0.1:5555")
    ap.add_argument("--proto", type=Path, default=Path.home() / "vla.cpp/src/serving/vla.proto")
    a = ap.parse_args()

    import zmq
    from phase_client import load_messages
    Req, Resp, F32_RGB_01 = load_messages(a.proto.expanduser())

    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 120000)
    sock.connect(a.addr)

    files = calib_files(a.calib)
    sel = select_rows(files, a.per_task)
    data = [np.load(f) for f in files]
    actions, diffs, t0 = [], [], time.time()
    for i, (t, r) in enumerate(sel):
        d = data[t]
        req = Req()
        req.request_id = i
        for view in d["images"][r]:                                   # [H, W, 3] uint8
            p = req.images.add()
            p.encoding = F32_RGB_01
            p.height, p.width = view.shape[:2]
            p.data = (view.astype(np.float32) / 255).tobytes()
        req.lang_tokens.extend(d["tokens"][r][:int(d["n_tokens"][r])].tolist())
        req.state.extend(d["state"][r].tolist())
        req.noise.extend(d["noise"][r].tolist())
        sock.send(req.SerializeToString())
        resp = Resp()
        resp.ParseFromString(sock.recv())
        if resp.error:
            raise RuntimeError(resp.error)
        act = np.array(resp.action_chunk, np.float32).reshape(resp.chunk_size, resp.action_dim)
        actions.append(act)
        diffs.append(np.abs(act[:, :7] - d["action"][r][:, :7]).mean())
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(sel)} requests, {time.time() - t0:.0f}s", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, task=np.array([t for t, _ in sel], np.int32), row=np.array([r for _, r in sel], np.int32),
             action=np.stack(actions))
    print(f"vs recorded (unpruned server, float images): mean |diff| over 7 dims {np.mean(diffs):.3e}, "
          f"max per-sample mean {np.max(diffs):.3e}")
    print(f"wrote {a.out} ({len(sel)} samples)")


if __name__ == "__main__":
    sys.exit(main())
