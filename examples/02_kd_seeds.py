"""Bài 2 (KD): repeat the key comparisons over several seeds.

Mentor feedback: "alpha=1 better than alpha=0" and "a smaller student benefits more" were inside
the noise of a single run. This script trains the same models as 02_knowledge_distillation.ipynb
(same architectures, loss, optimizer, data pipeline, 15 epochs) for several seeds and reports
mean and spread, plus paired differences (same seed = same init and same data order).

The teacher is trained once (seed 0) and cached, because the question is about student variance.
Each finished run is appended to the CSV, so an interrupted run resumes where it stopped.

Usage (from the repo root, conda env with torch + torchvision):
    python examples/02_kd_seeds.py                          # full run: 5 configs x 5 seeds
    python examples/02_kd_seeds.py --summary                # only print the summary of the CSV
    python examples/02_kd_seeds.py --epochs 1 --seeds 0 --configs student_a0 --out smoke.csv   # smoke test
"""
import argparse
import csv
import os
import statistics as st
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

BATCH_SIZE, LR, T_KD = 128, 0.05, 4.0


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Teacher(nn.Module):  # 1,147,722 params, same as the notebook
    def __init__(self, num_classes=10):
        super().__init__()
        c1, c2, c3 = 64, 128, 256
        self.features = nn.Sequential(
            conv_block(1, c1), conv_block(c1, c1), nn.MaxPool2d(2),
            conv_block(c1, c2), conv_block(c2, c2), nn.MaxPool2d(2),
            conv_block(c2, c3), conv_block(c3, c3), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class SmallCNN(nn.Module):
    """Student (16, 32, 32) = 14,458 params; TinyStudent (8, 16, 16) = 3,778 params."""

    def __init__(self, c1, c2, c3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, c1), nn.MaxPool2d(2),
            conv_block(c1, c2), nn.MaxPool2d(2),
            conv_block(c2, c3), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


STUDENTS = {"student": (16, 32, 32), "tiny": (8, 16, 16)}
# name -> (student kind, alpha). alpha = 0 is plain cross-entropy (no KD), alpha = 1 is teacher only.
CONFIGS = {
    "student_a0": ("student", 0.0),
    "student_a0.7": ("student", 0.7),
    "student_a1": ("student", 1.0),
    "tiny_a0": ("tiny", 0.0),
    "tiny_a0.7": ("tiny", 0.7),
}


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    ce_term = F.cross_entropy(student_logits, labels)
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    kd_term = F.kl_div(s, t, reduction="batchmean") * T**2
    return alpha * kd_term + (1 - alpha) * ce_term


def train(model, loss_fn, loader, device, epochs):
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), x, y).backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def count_errors(model, loader, device):
    model.eval()
    wrong = 0
    for x, y in loader:
        wrong += (model(x.to(device)).argmax(1) != y.to(device)).sum().item()
    return wrong


def load_done(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[int(r["seed"])] = int(r["errors"])
    print(f"\n{'config':<14}{'n':>3}{'mean errors':>13}{'std':>7}{'min':>6}{'max':>6}   per seed")
    for name in CONFIGS:
        if name not in by:
            continue
        v = [by[name][s] for s in sorted(by[name])]
        sd = st.stdev(v) if len(v) > 1 else float("nan")
        print(f"{name:<14}{len(v):>3}{st.mean(v):>13.1f}{sd:>7.1f}{min(v):>6}{max(v):>6}   {v}")

    def paired(a, b, label, relative=False):
        if a not in by or b not in by:
            return
        seeds = sorted(set(by[a]) & set(by[b]))
        if not seeds:
            return
        if relative:
            d = [100.0 * (by[a][s] - by[b][s]) / by[a][s] for s in seeds]
            unit = "%"
        else:
            d = [by[a][s] - by[b][s] for s in seeds]
            unit = " errors"
        sd = st.stdev(d) if len(d) > 1 else float("nan")
        wins = sum(x > 0 for x in d)
        print(f"{label:<44} mean {st.mean(d):6.1f}{unit}, std {sd:5.1f}, "
              f"{wins}/{len(d)} seeds > 0, per seed {[round(x, 1) for x in d]}")

    print("\nPaired by seed (positive = the second config has fewer errors):")
    paired("student_a0", "student_a0.7", "Student: no KD - KD (alpha 0.7)")
    paired("student_a0", "student_a1", "Student: alpha 0 - alpha 1")
    paired("student_a1", "student_a0.7", "Student: alpha 1 - alpha 0.7")
    paired("tiny_a0", "tiny_a0.7", "Tiny: no KD - KD (alpha 0.7)")
    paired("student_a0", "student_a0.7", "Student: relative error reduction by KD", relative=True)
    paired("tiny_a0", "tiny_a0.7", "Tiny: relative error reduction by KD", relative=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="examples/02_kd_seeds.csv")
    ap.add_argument("--teacher", default="data/kd_teacher_seed0.pt", help="cached teacher weights")
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summary", action="store_true", help="only print the summary of --out")
    args = ap.parse_args()

    if args.summary:
        summarize(load_done(args.out))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device: {name}, torch {torch.__version__}, epochs {args.epochs}")

    mean, std = (0.1307,), (0.3081,)
    train_tf = T.Compose([T.RandomCrop(28, padding=2), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_set = torchvision.datasets.MNIST(args.data, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.MNIST(args.data, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers)

    teacher = Teacher()
    if os.path.exists(args.teacher):
        teacher.load_state_dict(torch.load(args.teacher, map_location="cpu"))
        teacher.to(device)
        print(f"teacher loaded from {args.teacher}")
    else:
        t0 = time.time()
        torch.manual_seed(0)
        teacher = train(Teacher(), lambda out, x, y: F.cross_entropy(out, y), train_loader, device, args.epochs)
        os.makedirs(os.path.dirname(args.teacher) or ".", exist_ok=True)
        torch.save(teacher.state_dict(), args.teacher)
        print(f"teacher trained in {time.time() - t0:.0f} s, saved to {args.teacher}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"teacher errors: {count_errors(teacher, test_loader, device)} / {len(test_set)}")

    done = {(r["config"], r["seed"]) for r in load_done(args.out)}
    fields = ["config", "student", "alpha", "T", "seed", "epochs", "errors", "acc", "train_s", "device"]
    new_file = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for seed in [int(s) for s in args.seeds.split(",")]:
            for cfg in args.configs.split(","):
                if (cfg, str(seed)) in done:
                    print(f"skip {cfg} seed {seed} (already in {args.out})")
                    continue
                kind, alpha = CONFIGS[cfg]

                def loss_fn(out, x, y, alpha=alpha):
                    if alpha == 0.0:
                        return F.cross_entropy(out, y)
                    with torch.no_grad():
                        t_out = teacher(x)
                    return distillation_loss(out, t_out, y, T=T_KD, alpha=alpha)

                torch.manual_seed(seed)  # same seed -> same init and same data order for every config
                t0 = time.time()
                model = train(SmallCNN(*STUDENTS[kind]), loss_fn, train_loader, device, args.epochs)
                errors = count_errors(model, test_loader, device)
                dt = time.time() - t0
                w.writerow({"config": cfg, "student": kind, "alpha": alpha, "T": T_KD, "seed": seed,
                            "epochs": args.epochs, "errors": errors, "acc": f"{100 - errors / 100:.2f}",
                            "train_s": f"{dt:.0f}", "device": name})
                f.flush()
                print(f"{cfg:<14} seed {seed}: {errors} errors ({100 - errors / 100:.2f}%), {dt:.0f} s")

    summarize(load_done(args.out))


if __name__ == "__main__":  # needed on Windows because DataLoader workers re-import this file
    main()
