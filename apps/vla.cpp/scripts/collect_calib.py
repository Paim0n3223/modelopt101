"""Collect calibration inputs for CKA layer scoring by recording real LIBERO requests.

Runs LIBERO episodes exactly like vla.cpp's eval/client/run_sim_client_direct.py (same
client, adapter, env, n_action_steps=1) against a running vla-server, and records the
PredictRequest the client sends (2 preprocessed 512x512 views, prompt tokens, raw state,
noise) together with the action chunk the server returns. vla.cpp's client code is only
imported, not modified; no bytecode or files are written inside the vla.cpp clone.

VLA_FIXED_NOISE_SEED is set (default 0) so every request carries its noise vector; the
CKA script can then recompute the same forward pass in PyTorch and check it against the
recorded server action.

Usage (inside WSL, LIBERO venv, vla-server with the ORIGINAL model already listening):
  python collect_calib.py --out ~/profile/calib --tasks 0-9 --episodes 3 --every 20

Writes <out>/task_<t>.npz per task:
  images uint8 [N, V, 512, 512, 3] (value*255 rounded), tokens int32 [N, 48] (0-padded),
  n_tokens int32 [N], state float32 [N, S], noise float32 [N, 50*32],
  action float32 [N, 50, 32] (server output), episode/step int32 [N], success int8 [episodes]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

MAX_TOK = 48


class RecordingSocket:
    """Proxy around the client's ZMQ socket that remembers each (request, reply) pair."""

    def __init__(self, sock):
        self._sock = sock
        self._last = None
        self.pairs = []

    def send(self, data, *args, **kwargs):
        self._last = data
        return self._sock.send(data, *args, **kwargs)

    def recv(self, *args, **kwargs):
        body = self._sock.recv(*args, **kwargs)
        self.pairs.append((self._last, body))
        return body

    def __getattr__(self, name):
        return getattr(self._sock, name)


def parse_tasks(spec):
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", default="0-9", help='e.g. "0-9" or "0,3,5"')
    ap.add_argument("--episodes", type=int, default=3, help="episodes per task")
    ap.add_argument("--every", type=int, default=20, help="keep one request every N predictions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--addr", default="tcp://127.0.0.1:5555")
    ap.add_argument("--vla-dir", type=Path, default=Path.home() / "vla.cpp")
    a = ap.parse_args()

    os.environ.setdefault("VLA_FIXED_NOISE_SEED", "0")
    sys.dont_write_bytecode = True                      # no __pycache__ inside the vla.cpp clone
    sys.path.insert(0, str(a.vla_dir.expanduser() / "eval"))
    import gymnasium as gym
    import sim.libero  # noqa: F401  registers the LIBERO gymnasium envs
    from client.adapters import LeRobotPipelineAdapter
    from client.vla_cpp_client import VlaCppClient

    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    inner = VlaCppClient(vla_addr=a.addr, arch="smolvla", n_action_steps=1)
    rec = RecordingSocket(inner.sock)
    inner.sock = rec
    client = LeRobotPipelineAdapter(client=inner)
    pb = inner.pb

    for task in parse_tasks(a.tasks):
        env = gym.make(f"{a.suite}/task_{task}", seed=a.seed, output_video_dir=None)
        rows = {k: [] for k in ("images", "tokens", "n_tokens", "state", "noise", "action", "episode", "step")}
        successes = []
        t_task = time.time()
        for ep in range(a.episodes):
            client.reset()
            obs, info = env.reset()
            n_pred, steps, reward = 0, 0, 0.0
            while True:
                rec.pairs.clear()
                action = client.get_action(obs)
                for req_b, resp_b in rec.pairs:
                    req = pb.PredictRequest()
                    req.ParseFromString(req_b)
                    if not req.noise:                   # the client's one-off noise-length probe
                        continue
                    if n_pred % a.every == 0:
                        resp = pb.PredictResponse()
                        resp.ParseFromString(resp_b)
                        views = [np.frombuffer(im.data, np.float32).reshape(im.height, im.width, 3)
                                 for im in req.images]
                        rows["images"].append(np.clip(np.rint(np.stack(views) * 255), 0, 255).astype(np.uint8))
                        tok = np.zeros(MAX_TOK, np.int32)
                        tok[:len(req.lang_tokens)] = req.lang_tokens
                        rows["tokens"].append(tok)
                        rows["n_tokens"].append(len(req.lang_tokens))
                        rows["state"].append(np.array(req.state, np.float32))
                        rows["noise"].append(np.array(req.noise, np.float32))
                        rows["action"].append(np.array(resp.action_chunk, np.float32)
                                              .reshape(resp.chunk_size, resp.action_dim))
                        rows["episode"].append(ep)
                        rows["step"].append(steps)
                    n_pred += 1
                try:
                    obs, reward, done, truncated, info = env.step(action)
                except ValueError as e:                 # same handling as run_sim_client_direct.py
                    if "terminated episode" not in str(e):
                        raise
                    done, truncated = True, False
                steps += 1
                if done or truncated:
                    break
            ok = int(bool(info.get("is_success", 0)))
            successes.append(ok)
            print(f"task {task} episode {ep}: success={ok} steps={steps} kept={len(rows['step'])}", flush=True)
        env.close()
        np.savez(out / f"task_{task}.npz",
                 images=np.stack(rows["images"]), tokens=np.stack(rows["tokens"]),
                 n_tokens=np.array(rows["n_tokens"], np.int32), state=np.stack(rows["state"]),
                 noise=np.stack(rows["noise"]), action=np.stack(rows["action"]),
                 episode=np.array(rows["episode"], np.int32), step=np.array(rows["step"], np.int32),
                 success=np.array(successes, np.int8))
        print(f"== task {task}: {len(rows['step'])} samples, success {sum(successes)}/{a.episodes}, "
              f"{time.time() - t_task:.0f}s -> {out / f'task_{task}.npz'}", flush=True)


if __name__ == "__main__":
    main()
