"""Per-phase latency of a running vla-server (SmolVLA).

Sends the same request N times (2 views 512x512, a LIBERO prompt, zero state,
fixed noise) and reads the timings the server returns in each response:
total / vision / inference / prefill / denoise (prefill and denoise are only
filled when the server runs with --timing-detail phase).

Usage (inside WSL, with vla-server already listening):
  python phase_client.py --label bf16_none --out results/profile/phase/bf16_none.csv

Writes one CSV row per timed request and prints a median / p90 summary.
The request content does not change the compute (fixed shapes), so a synthetic
image is used. Needs numpy, pyzmq, protobuf, transformers (~/venvs/vla-tok has them)
and protoc on PATH.
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import zmq
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

S = 512                                  # SmolVLA image size
N_SUFFIX, MAX_ACTION, MAX_STATE = 50, 32, 32
TOKENIZER = "HuggingFaceTB/SmolVLM2-500M-Instruct"
FIELDS = ["total", "vision", "inference", "prefill", "denoise", "other", "roundtrip"]


def load_messages(proto):
    """Build PredictRequest/PredictResponse classes from vla.proto via a descriptor set
    (independent of the protoc / protobuf runtime versions). Writes only to a temp dir."""
    with tempfile.TemporaryDirectory(prefix="vla-desc-") as tmp:
        desc = Path(tmp) / "vla.desc"
        subprocess.check_call(["protoc", f"--proto_path={proto.parent}",
                               f"--descriptor_set_out={desc}", str(proto)])
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(desc.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for fd in fds.file:
        pool.Add(fd)
    get = message_factory.GetMessageClass
    return (get(pool.FindMessageTypeByName("vla.PredictRequest")),
            get(pool.FindMessageTypeByName("vla.PredictResponse")),
            pool.FindEnumTypeByName("vla.Image.Encoding").values_by_name["F32_RGB_01"].number)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="tcp://127.0.0.1:5555")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n", type=int, default=50, help="timed requests")
    ap.add_argument("--prompt", default="pick up the alphabet soup and place it in the basket")
    ap.add_argument("--proto", type=Path,
                    default=Path(os.environ.get("VLA_DIR", Path.home() / "vla.cpp")) / "src/serving/vla.proto")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True, help="CSV, one row per timed request")
    a = ap.parse_args()

    Req, Resp, F32_RGB_01 = load_messages(a.proto)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    text = a.prompt if a.prompt.endswith("\n") else a.prompt + "\n"   # same as the LIBERO client
    tokens = tok(text, truncation=True, max_length=48)["input_ids"]

    rng = np.random.default_rng(0)
    images = [rng.random((S, S, 3), dtype=np.float32) for _ in range(a.views)]
    noise = rng.standard_normal(N_SUFFIX * MAX_ACTION).astype(np.float32)

    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 120000)
    sock.connect(a.addr)

    def predict(rid):
        req = Req()
        req.request_id = rid
        for im in images:
            p = req.images.add()
            p.encoding = F32_RGB_01
            p.height, p.width = S, S
            p.data = im.tobytes()
        req.lang_tokens.extend(tokens)
        req.state.extend([0.0] * MAX_STATE)
        req.noise.extend(noise.tolist())
        body = req.SerializeToString()
        t0 = time.perf_counter()
        sock.send(body)
        resp = Resp()
        resp.ParseFromString(sock.recv())
        rt = (time.perf_counter() - t0) * 1000
        if resp.error:
            raise RuntimeError(resp.error)
        return {"total": resp.latency_ms_total, "vision": resp.latency_ms_vision,
                "inference": resp.latency_ms_inference, "prefill": resp.latency_ms_prefill,
                "denoise": resp.latency_ms_denoise,
                "other": resp.latency_ms_total - resp.latency_ms_vision - resp.latency_ms_inference,
                "roundtrip": rt}

    print(f"{a.label}: {len(tokens)} prompt tokens, {a.views} views, {a.warmup} warmup + {a.n} timed",
          flush=True)
    for i in range(a.warmup):
        predict(i)
    rows = [predict(a.warmup + i) for i in range(a.n)]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i"] + FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"i": i, **{k: f"{r[k]:.3f}" for k in FIELDS}})

    def p90(v):
        return sorted(v)[max(0, int(round(0.9 * len(v))) - 1)]
    print(f"{'ms':10s} {'median':>8s} {'p90':>8s}")
    for k in FIELDS:
        v = [r[k] for r in rows]
        print(f"{k:10s} {statistics.median(v):8.1f} {p90(v):8.1f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
