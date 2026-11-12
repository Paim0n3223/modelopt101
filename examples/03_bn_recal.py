"""Bài 3 (pruning): recompute BatchNorm statistics after structured pruning, without training.

Mentor feedback: 9.74% right after cutting means the model predicts one class for every image
(class 8 has exactly 974 test images). L1 may be no better than random because BatchNorm
renormalises each channel. Try recomputing the BN running statistics after the cut (no gradient,
no labels) and measure how much accuracy comes back.

For every dense seed, ratio and criterion (L1 of the conv weight, |BN gamma|, random) the script
records accuracy and the share of the most predicted class, before and after BN recalibration.
As a reference it also fine-tunes the L1 cut at ratio 0.5 for 4 epochs, like the notebook.

Same model, data pipeline and training recipe as 03_pruning.ipynb (12 epochs dense, fine-tune
4 epochs at LR * 0.2). Dense weights are cached in data/, finished rows are appended to the CSV,
so an interrupted run resumes where it stopped.

Usage (from the repo root, conda env with torch + torchvision):
    python examples/03_bn_recal.py                       # full run
    python examples/03_bn_recal.py --summary             # only print the summary of the CSV
    python examples/03_bn_recal.py --dense-seeds 0 --epochs 1 --ratios 0.5 --out smoke.csv --cache data/smoke   # smoke test
"""
import argparse
import copy
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

BATCH_SIZE, LR = 128, 0.05
CONV_IDX = [0, 1, 3, 4, 6, 7]


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Net(nn.Module):  # same as the notebook: default widths -> 1,147,722 params
    def __init__(self, widths=(64, 64, 128, 128, 256, 256), num_classes=10):
        super().__init__()
        w = widths
        self.features = nn.Sequential(
            conv_block(1, w[0]), conv_block(w[0], w[1]), nn.MaxPool2d(2),
            conv_block(w[1], w[2]), conv_block(w[2], w[3]), nn.MaxPool2d(2),
            conv_block(w[3], w[4]), conv_block(w[4], w[5]), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w[5], num_classes))
        self.widths = tuple(widths)

    def forward(self, x):
        return self.head(self.features(x))


def conv_blocks(model):
    return [model.features[i] for i in CONV_IDX]


def structured_prune(model, ratio, criterion="l1", seed=0):
    """Same as structured_prune_c in reports/03_pruning.md: 'l1', 'bn' (|gamma|) or 'random'."""
    g = torch.Generator().manual_seed(seed)
    blocks = conv_blocks(model)
    keep = []
    for blk in blocks:
        n = blk[0].weight.shape[0]
        k = max(1, round(n * (1 - ratio)))
        if criterion == "l1":
            idx = torch.topk(blk[0].weight.data.abs().sum(dim=(1, 2, 3)), k).indices.sort().values
        elif criterion == "bn":
            idx = torch.topk(blk[1].weight.data.abs(), k).indices.sort().values
        else:
            idx = torch.randperm(n, generator=g)[:k].sort().values
        keep.append(idx.to(blk[0].weight.device))

    new = Net(widths=tuple(len(k) for k in keep)).to(blocks[0][0].weight.device).eval()
    prev = torch.tensor([0], device=blocks[0][0].weight.device)
    for ob, nb, idx in zip(blocks, conv_blocks(new), keep):
        nb[0].weight.data = ob[0].weight.data[idx][:, prev, :, :].clone()
        nb[1].weight.data = ob[1].weight.data[idx].clone()
        nb[1].bias.data = ob[1].bias.data[idx].clone()
        nb[1].running_mean = ob[1].running_mean[idx].clone()
        nb[1].running_var = ob[1].running_var[idx].clone()
        prev = idx
    new.head[2].weight.data = model.head[2].weight.data[:, prev].clone()
    new.head[2].bias.data = model.head[2].bias.data.clone()
    return new


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches, device):
    """Forget the old running mean/var and recompute them on training images (no gradient, no labels)."""
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all calibration batches
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(device))
    return model.eval()


@torch.no_grad()
def evaluate(model, loader, device):
    """Accuracy (%) and the share (%) of the most predicted class."""
    model.eval()
    correct, counts = 0, torch.zeros(10, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        counts += torch.bincount(pred, minlength=10)
    n = counts.sum().item()
    top = counts.argmax().item()
    return 100.0 * correct / n, top, 100.0 * counts[top].item() / n


def train(model, loader, device, epochs, lr):
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
    return model.eval()


FIELDS = ["dense_seed", "ratio", "criterion", "crit_seed", "acc_before", "top_before", "top_share_before",
          "acc_after", "top_after", "top_share_after", "calib_batches", "device"]


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((float(r["ratio"]), r["criterion"]), []).append(r)
    print(f"\n{'ratio':>5} {'criterion':<12}{'n':>3}  {'before: acc':>11} {'top class %':>11}"
          f"   {'after BN recal: acc':>19} {'top class %':>11}")
    for (ratio, crit) in sorted(groups):
        g = groups[(ratio, crit)]
        b = [float(r["acc_before"]) for r in g]
        a = [float(r["acc_after"]) for r in g]
        tb = [float(r["top_share_before"]) for r in g]
        ta = [float(r["top_share_after"]) for r in g]
        rng = lambda v: f"{st.mean(v):6.2f} ({min(v):.2f}-{max(v):.2f})" if len(v) > 1 else f"{v[0]:6.2f}"
        print(f"{ratio:>5} {crit:<12}{len(g):>3}  {rng(b):>24} {st.mean(tb):>6.1f}   {rng(a):>24} {st.mean(ta):>6.1f}")
    print("\nbefore = right after the cut (old BN statistics); after = BN statistics recomputed, no training.")
    print("For 'l1_finetune', before = right after the cut, after = after 4 epochs of fine-tuning.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-seeds", default="0,1,2")
    ap.add_argument("--ratios", default="0.1,0.2,0.3,0.5")
    ap.add_argument("--random-seeds", default="0,1,2,3,4")
    ap.add_argument("--calib-batches", type=int, default=100, help="batches of 128 training images")
    ap.add_argument("--epochs", type=int, default=12, help="dense training epochs")
    ap.add_argument("--finetune-epochs", type=int, default=4)
    ap.add_argument("--out", default="examples/03_bn_recal.csv")
    ap.add_argument("--cache", default="data/bn_dense", help="prefix of cached dense weights")
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summarize(load_rows(args.out))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device: {dev_name}, torch {torch.__version__}")

    mean, std = (0.1307,), (0.3081,)
    train_tf = T.Compose([T.RandomCrop(28, padding=2), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_set = torchvision.datasets.MNIST(args.data, train=True, download=True, transform=train_tf)
    calib_set = torchvision.datasets.MNIST(args.data, train=True, download=True, transform=test_tf)
    test_set = torchvision.datasets.MNIST(args.data, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers)

    done = {(r["dense_seed"], r["ratio"], r["criterion"], r["crit_seed"]) for r in load_rows(args.out)}
    new_file = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    def write(row):
        row.update(calib_batches=args.calib_batches, device=dev_name)
        w.writerow(row)
        f.flush()
        print(f"  seed {row['dense_seed']} ratio {row['ratio']} {row['criterion']:<12} crit_seed {row['crit_seed']}: "
              f"{row['acc_before']}% (top {row['top_share_before']}%) -> {row['acc_after']}% (top {row['top_share_after']}%)")

    for ds in [int(s) for s in args.dense_seeds.split(",")]:
        path = f"{args.cache}_seed{ds}.pt"
        model = Net()
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.to(device).eval()
            print(f"dense seed {ds}: loaded {path}")
        else:
            t0 = time.time()
            torch.manual_seed(ds)
            model = train(Net(), train_loader, device, args.epochs, LR)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(model.state_dict(), path)
            print(f"dense seed {ds}: trained in {time.time() - t0:.0f} s, saved to {path}")

        # calibration batches come from a fixed-seed shuffle of the training set (no augmentation)
        g = torch.Generator().manual_seed(1000 + ds)
        calib_loader = DataLoader(calib_set, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0)

        jobs = [("0.0", "dense", 0)]
        for r in args.ratios.split(","):
            jobs += [(r, "l1", 0), (r, "bn", 0)] + [(r, "random", s) for s in map(int, args.random_seeds.split(","))]
        for r, crit, cs in jobs:
            if (str(ds), r, crit, str(cs)) in done:
                continue
            m = model if crit == "dense" else structured_prune(model, float(r), crit, cs)
            acc_b, top_b, share_b = evaluate(m, test_loader, device)
            acc_a, top_a, share_a = evaluate(recalibrate_bn(m, calib_loader, args.calib_batches, device), test_loader, device)
            write(dict(dense_seed=ds, ratio=r, criterion=crit, crit_seed=cs,
                       acc_before=f"{acc_b:.2f}", top_before=top_b, top_share_before=f"{share_b:.1f}",
                       acc_after=f"{acc_a:.2f}", top_after=top_a, top_share_after=f"{share_a:.1f}"))

        # reference: the notebook's recovery path (L1, ratio 0.5, fine-tune 4 epochs at LR * 0.2)
        if args.finetune_epochs > 0 and (str(ds), "0.5", "l1_finetune", "0") not in done:
            m = structured_prune(model, 0.5, "l1")
            acc_b, top_b, share_b = evaluate(m, test_loader, device)
            torch.manual_seed(ds)
            t0 = time.time()
            m = train(m, train_loader, device, args.finetune_epochs, LR * 0.2)
            acc_a, top_a, share_a = evaluate(m, test_loader, device)
            print(f"  fine-tune took {time.time() - t0:.0f} s")
            write(dict(dense_seed=ds, ratio="0.5", criterion="l1_finetune", crit_seed=0,
                       acc_before=f"{acc_b:.2f}", top_before=top_b, top_share_before=f"{share_b:.1f}",
                       acc_after=f"{acc_a:.2f}", top_after=top_a, top_share_after=f"{share_a:.1f}"))

    f.close()
    summarize(load_rows(args.out))


if __name__ == "__main__":  # needed on Windows because DataLoader workers re-import this file
    main()
