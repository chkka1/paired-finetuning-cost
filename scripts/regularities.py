# -*- coding: utf-8 -*-
"""跨五个数据集的一致性规律检验(PACS / Organ / Digits / OfficeHome / CORe50)。

对每个数据集检验以下候选规律(全部用主档 3e-4 与 1e-3 结果，10 种子)：
  R1 ΔIntra 的增量解释力：R²(ΔF+ΔIntra) − R²(ΔF)(域外观测点，按域×轮取种子均值)
  R2 域内与域外不同向：末轮域内 Δacc 的符号与各域外域符号不一致的域数
  R3 强度越高域外越差：1e-3 相对 3e-4 在域外域上更差的域数占比
  R4 最近类心与岭回归同向：acc_nc 与 acc_ridge 的 Pearson r
  R5 初始距离越大越吃亏：域外域在**未微调**时的初始 Fréchet 距离 vs 末轮 Δacc 的相关(按域×种子)
  R6 两种距离度量同向：合并斜率用 ΔMMD 与用 ΔFréchet 的符号是否一致
输出：终端表 + results/regularities.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CASES = [("pacs", 128), ("organ", 64), ("digits", 64), ("officehome", 128), ("core50", 64)]
TAGS = {"": "3e-4", "_strong": "1e-3"}


def rows_of(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def r2_fit(x, y):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2_fit2(x1, x2, y):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x1, float), np.asarray(x2, float)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


def slope_signed(x, y):
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(b[1])


out = {}
print("=" * 118)
print("跨五个数据集的一致性规律(10 种子；主档 3e-4；Δacc 单位 pp)")
print("=" * 118)
print("  %-11s %8s %8s %10s %10s %12s %10s %9s" % (
    "数据集", "R1 ΔR²", "R2 反号", "R3 强更差", "R4 r(nc,rg)", "R5 r(初距,Δ)", "R6 mmd/fr", "域外 n"))
for case, size in CASES:
    base_tag = RES / ("%s_resnet18_s%d.csv" % (case, size))
    R = rows_of(base_tag)
    if not R:
        print("  %-11s (缺少结果文件)" % case)
        continue
    ep = max(int(r["epoch"]) for r in R)
    bas = {(int(r["seed"]), r["domain"]): (float(r["acc_ridge"]), float(r["fr_dist"]),
                                           float(r["intra"]), float(r["mmd"]))
           for r in R if int(r["epoch"]) == 0}
    doms = sorted({r["domain"] for r in R})
    ood = [d for d in doms if "域内" not in d]
    ind = [d for d in doms if "域内" in d][0]
    # 域外观测点(域×轮，种子均值)
    pts = {}
    for r in R:
        e, sd = int(r["epoch"]), int(r["seed"])
        if e == 0 or "域内" in r["domain"]:
            continue
        a0, f0, i0, m0 = bas[(sd, r["domain"])]
        pts.setdefault((r["domain"], e), {})[sd] = (100 * (float(r["acc_ridge"]) - a0),
                                                   float(r["fr_dist"]) - f0,
                                                   float(r["intra"]) - i0,
                                                   float(r["mmd"]) - m0)
    xs = [np.mean([v[1] for v in pts[k].values()]) for k in sorted(pts)]
    xin = [np.mean([v[2] for v in pts[k].values()]) for k in sorted(pts)]
    xm = [np.mean([v[3] for v in pts[k].values()]) for k in sorted(pts)]
    ys = [np.mean([v[0] for v in pts[k].values()]) for k in sorted(pts)]
    dR2 = r2_fit2(xs, xin, ys) - r2_fit(xs, ys)
    # 末轮各域 Δacc
    def final_delta(tag):
        RR = rows_of(RES / ("%s_resnet18_s%d%s.csv" % (case, size, tag)))
        if not RR:
            return None
        b2 = {(int(r["seed"]), r["domain"]): float(r["acc_ridge"]) for r in RR if int(r["epoch"]) == 0}
        e = max(int(r["epoch"]) for r in RR)
        return {d: float(np.mean([100 * (float(r["acc_ridge"]) - b2[(int(r["seed"]), d)])
                                  for r in RR if int(r["epoch"]) == e and r["domain"] == d]))
                for d in doms}
    d3, ds = final_delta(""), final_delta("_strong")
    in_sign = np.sign(d3[ind]) if d3 else np.nan
    mismatch = sum(1 for d in ood if d3 and np.sign(d3[d]) != in_sign and abs(d3[d]) > 0.5)
    worse = sum(1 for d in ood if d3 and ds and ds[d] < d3[d])
    # R4：最近类心 vs 岭回归
    nc = [float(r["acc_nc"]) for r in R]
    rg = [float(r["acc_ridge"]) for r in R]
    r4 = float(np.corrcoef(nc, rg)[0, 1])
    # R5：初始距离 vs 末轮 Δacc(域×种子)
    x5, y5 = [], []
    for d in ood:
        for sd in sorted({int(r["seed"]) for r in R}):
            x5.append(bas[(sd, d)][1])
            y5.append(100 * (float([r for r in R if int(r["seed"]) == sd and int(r["epoch"]) == ep
                                    and r["domain"] == d][0]["acc_ridge"]) - bas[(sd, d)][0]))
    r5 = float(np.corrcoef(x5, y5)[0, 1]) if len(x5) > 2 else float("nan")
    s_mmd, s_fr = slope_signed(xm, ys), slope_signed(xs, ys)
    out[case] = dict(dR2_intra=round(float(dR2), 3), mismatch=mismatch, n_ood=len(ood),
                     strong_worse=worse, r_nc_rg=round(r4, 3), r_init=round(r5, 3),
                     slope_mmd=round(s_mmd, 3), slope_fr=round(s_fr, 3),
                     same_sign_measure=bool(np.sign(s_mmd) == np.sign(s_fr)))
    print("  %-11s %+8.3f %6d/%-3d %6d/%-3d %10.3f %12.3f %10s %9d" % (
        case, dR2, mismatch, len(ood), worse, len(ood), r4, r5,
        "同号" if np.sign(s_mmd) == np.sign(s_fr) else "反号", len(ood)))
(RES / "regularities.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n已写出:", RES / "regularities.json")
