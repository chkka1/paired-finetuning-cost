# -*- coding: utf-8 -*-
"""微调的域内收益 vs 域外代价：配对实验(PACS / Organ / Digits / OfficeHome)。

核心协议(保证域内与域外可比，避免"不同口径相减")：
  1. 只在源域训练集(splits==0)上微调主干；微调期间挂线性分类头，评估时丢弃。
  2. 评估一律使用**同一套在源域训练集特征上重新拟合的线性分类器**(岭回归 one-hot)
     与最近类心(cosine)，因此每次评估只有"表示"在变，分类规则不变。
  3. 域内 = 源域测试集；域外 = 其余域的测试集(固定抽样、固定顺序)。
  4. 逐 epoch 记录，得到剂量-反应曲线；每个 seed 独立重复。
  5. 同步记录特征空间诊断(域间 Fréchet 距离、线性 MMD、类内紧致度)。

输出：results/<case>_<backbone>_s<size>.csv(长表，逐行增量写入，可断点续跑)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "cases"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

TEST_CAP = 5000          # 默认每域测试集抽样上限(可用 --test-cap 覆盖)
RIDGE_LAMBDA = 1.0


# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_case(name: str):
    z = np.load(CASES / (name + ".npz"), allow_pickle=True)
    return dict(images=z["images"], labels=z["labels"], domains=z["domains"],
                splits=z["splits"], domain_names=[str(x) for x in z["domain_names"]])


def stratified_cap(idx: np.ndarray, labels: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    if len(idx) <= cap:
        return idx
    rng = np.random.RandomState(seed)
    out = []
    per = max(1, cap // len(set(labels[idx].tolist())))
    for c in sorted(set(labels[idx].tolist())):
        sub = idx[labels[idx] == c]
        out.append(rng.choice(sub, min(per, len(sub)), replace=False))
    return np.sort(np.concatenate(out))


def to_tensor(images: np.ndarray, size: int) -> torch.Tensor:
    """uint8 [N,h,w,3] -> float tensor [N,3,size,size]，ImageNet 归一化。"""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    return x


# --------------------------------------------------------------------------- #
def build_backbone(name: str, n_classes: int):
    import torchvision
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity(); dim = 512
    elif name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity(); dim = 2048
    elif name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity(); dim = 1280
    elif name == "vit_b_16":
        # ViT-B/16 的位置编码按 224×224(14×14 patch)预训练，故输入尺寸须为 224
        m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity(); dim = 768
    else:
        raise ValueError(name)
    head = nn.Linear(dim, n_classes)
    return m, head, dim


@torch.no_grad()
def extract(model, x: torch.Tensor, device, batch=256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch].to(device)).float().cpu())
    return torch.cat(out).numpy()


def l2norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-8)


def nearest_centroid(Xtr, ytr, Xte, n_classes):
    C = np.zeros((n_classes, Xtr.shape[1]), np.float32)
    for c in range(n_classes):
        m = ytr == c
        if m.any():
            C[c] = Xtr[m].mean(0)
    C = l2norm(C)
    return (l2norm(Xte) @ C.T).argmax(1)


def ridge_probe(Xtr, ytr, Xte, n_classes, lam=RIDGE_LAMBDA):
    X = l2norm(Xtr); Xte = l2norm(Xte)
    X = np.hstack([X, np.ones((len(X), 1), np.float32)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    Y = np.zeros((len(X), n_classes), np.float32); Y[np.arange(len(X)), ytr] = 1.0
    d = X.shape[1]
    W = np.linalg.solve(X.T @ X + lam * np.eye(d, dtype=np.float32), X.T @ Y)
    return (Xte @ W).argmax(1)


def frechet(A, B):
    """Fréchet 距离(均值+协方差)，用对角协方差近似以稳定小样本估计。"""
    ma, mb = A.mean(0), B.mean(0)
    va, vb = A.var(0), B.var(0)
    return float(np.sum((ma - mb) ** 2) + np.sum((np.sqrt(va) - np.sqrt(vb)) ** 2))


def linear_mmd(A, B):
    return float(np.sum((A.mean(0) - B.mean(0)) ** 2))


def intra_compactness(X, y, n_classes):
    """类内平均 cosine 距离(到类心)，越小越紧致。"""
    Xn = l2norm(X)
    vals = []
    for c in range(n_classes):
        m = y == c
        if m.sum() < 2:
            continue
        ctr = l2norm(Xn[m].mean(0, keepdims=True))
        vals.append(float((1.0 - (Xn[m] @ ctr.T)).mean()))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
def evaluate(model, device, n_classes, packs, src_stats):
    """packs: {domain_name: (x_tensor, y_np)}；分类器一律在源域训练集特征上拟合。"""
    Xtr, ytr = src_stats["Xtr"], src_stats["ytr"]
    rows = []
    for dname, (xt, y) in packs.items():
        X = extract(model, xt, device)
        pred_nc = nearest_centroid(Xtr, ytr, X, n_classes)
        pred_rg = ridge_probe(Xtr, ytr, X, n_classes, RIDGE_LAMBDA)
        rows.append(dict(
            domain=dname, n=int(len(y)),
            acc_nc=float((pred_nc == y).mean()), acc_ridge=float((pred_rg == y).mean()),
            fr_dist=frechet(Xtr, X), mmd=linear_mmd(Xtr, X),
            intra=intra_compactness(X, y, n_classes),
            per_class={int(c): float((pred_rg[y == c] == c).mean()) for c in sorted(set(y.tolist()))},
        ))
    return rows


def run(args):
    case = load_case(args.case)
    images, labels, domains, splits = case["images"], case["labels"], case["domains"], case["splits"]
    names = case["domain_names"]
    n_classes = len(set(labels.tolist()))
    if args.src is not None:
        src = names.index(args.src)
    else:  # 自动识别源域＝有训练集(splits==0)的那个域
        counts = [int(((domains == d) & (splits == 0)).sum()) for d in range(len(names))]
        src = int(np.argmax(counts))
        print("auto src = {} (train={})".format(names[src], counts[src]), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定抽样：源域 train/val/test 与各域 test
    rng = np.random.RandomState(args.data_seed)
    src_tr = np.where((domains == src) & (splits == 0))[0]
    src_val = np.where((domains == src) & (splits == 1))[0]
    src_te = np.where((domains == src) & (splits == 2))[0]
    if args.max_train and len(src_tr) > args.max_train:
        src_tr = stratified_cap(src_tr, labels, args.max_train, args.data_seed)
    packs = {}
    src_te_capped = stratified_cap(src_te, labels, args.test_cap, args.data_seed)
    packs[names[src] + "(域内)"] = (to_tensor(images[src_te_capped], args.size), labels[src_te_capped])
    for d in range(len(names)):
        if d == src:
            continue
        idx = np.where((domains == d) & (splits == 2))[0]
        idx = stratified_cap(idx, labels, args.test_cap, args.data_seed)
        packs[names[d] + "(域外)"] = (to_tensor(images[idx], args.size), labels[idx])

    out_csv = RESULTS / "{}_{}_s{}{}.csv".format(args.case, args.backbone, args.size, args.tag)
    done = set()
    if out_csv.exists():
        import csv as _csv
        with out_csv.open(encoding="utf-8-sig") as fh:
            for r in _csv.DictReader(fh):
                done.add((int(r["seed"]), int(r["epoch"])))

    print("[{}] src={} classes={} in-domain test={} ".format(
        args.case, names[src], n_classes, len(src_te))
        + " ".join("{}={}".format(k, len(v[1])) for k, v in packs.items()), flush=True)

    header = ("case,backbone,size,seed,epoch,domain,role,n,acc_nc,acc_ridge,"
              "fr_dist,mmd,intra,secs")
    new_file = not out_csv.exists()
    fh = out_csv.open("a", encoding="utf-8-sig", newline="")
    if new_file:
        fh.write(header + "\n")

    for seed in args.seeds:
        set_seed(seed)
        torch.manual_seed(seed)
        model, head, dim = build_backbone(args.backbone, n_classes)
        model.to(device); head.to(device)
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
        tr_x = to_tensor(images[src_tr], args.size)
        tr_y = torch.from_numpy(labels[src_tr]).long()
        ds = TensorDataset(tr_x, tr_y)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False,
                        num_workers=0, generator=torch.Generator().manual_seed(seed))
        aug = args.aug

        # epoch 0 = 未微调基线
        if (seed, 0) not in done:
            Xtr = extract(model, tr_x, device, args.batch)
            src_stats = dict(Xtr=Xtr, ytr=labels[src_tr])
            t0 = time.time()
            for r in evaluate(model, device, n_classes, packs, src_stats):
                fh.write("{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f}\n".format(
                    args.case, args.backbone, args.size, seed, 0, r["domain"],
                    "in" if "(域内)" in r["domain"] else "out", r["n"],
                    r["acc_nc"], r["acc_ridge"], r["fr_dist"], r["mmd"], r["intra"], time.time() - t0))
            fh.flush()
            print("  seed {} epoch 0 done".format(seed), flush=True)

        for ep in range(1, args.epochs + 1):
            model.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                if aug:
                    # 简单增强：随机水平翻转 + 随机裁剪缩放
                    if random.random() < 0.5:
                        xb = torch.flip(xb, dims=[3])
                    if random.random() < 0.5:
                        s = random.uniform(0.75, 1.0)
                        sz = max(32, int(args.size * s))
                        xb = F.interpolate(xb, size=(sz, sz), mode="bilinear", align_corners=False)
                        oy = random.randint(0, sz - args.size) if sz > args.size else 0
                        ox = random.randint(0, sz - args.size) if sz > args.size else 0
                        xb = xb[:, :, oy:oy + args.size, ox:ox + args.size]
                        if xb.shape[-1] != args.size:
                            xb = F.interpolate(xb, size=(args.size, args.size), mode="bilinear",
                                               align_corners=False)
                opt.zero_grad(set_to_none=True)
                logits = head(model(xb))
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
            sched.step()

            if ep in args.eval_epochs and (seed, ep) not in done:
                t0 = time.time()
                Xtr = extract(model, tr_x, device, args.batch)
                src_stats = dict(Xtr=Xtr, ytr=labels[src_tr])
                for r in evaluate(model, device, n_classes, packs, src_stats):
                    fh.write("{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f}\n".format(
                        args.case, args.backbone, args.size, seed, ep, r["domain"],
                        "in" if "(域内)" in r["domain"] else "out", r["n"],
                        r["acc_nc"], r["acc_ridge"], r["fr_dist"], r["mmd"], r["intra"],
                        time.time() - t0))
                fh.flush()
                print("  seed {} epoch {} done ({:.0f}s)".format(seed, ep, time.time() - t0), flush=True)
    fh.close()
    print("saved:", out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["pacs", "organ", "digits", "officehome", "core50", "core50all"])
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--src", default=None, help="源域名(默认取第一个域)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eval-epochs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--test-cap", type=int, default=5000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(args)


if __name__ == "__main__":
    main()
