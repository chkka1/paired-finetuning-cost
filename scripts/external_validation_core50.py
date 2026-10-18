# -*- coding: utf-8 -*-
"""CORe50 外部验证分析(预注册第 11 节口径)。

  - P1：域外观测点上 Δacc 对 ΔFréchet 的标准化回归斜率 < 0；
  - P2：加入 ΔIntra 后 R² 上升 ≥ 0.1；
  - F1：斜率 ≥ 0 且 95% CI 不含 0；
  - 统计：按(域 × 轮数)对 10 个种子取均值后回归；逐域检验做 Holm 校正。
输出：results/external_validation2.json(供成稿使用)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CASE, SIZE = "core50", 64
CONFIGS = [("lr=3e-4(预注册主检验)", "", "3e-4"),
           ("lr=1e-3", "_strong", "1e-3"),
           ("lr=1e-4", "_lr1e4", "1e-4")]
DOM_ZH = {"s1": "s1(源域)", "s3": "s3", "s7": "s7", "s10": "s10"}


def load(tag: str):
    p = RES / ("%s_resnet18_s%d%s.csv" % (CASE, SIZE, tag))
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    base = {(int(r["seed"]), r["domain"]): (float(r["acc_ridge"]), float(r["fr_dist"]), float(r["intra"]))
            for r in rows if int(r["epoch"]) == 0}
    pts = {}
    for r in rows:
        e, sd = int(r["epoch"]), int(r["seed"])
        if e == 0:
            continue
        k = (sd, r["domain"])
        if k not in base:
            continue
        a0, f0, i0 = base[k]
        pts.setdefault((r["domain"], e), {})[sd] = (100 * (float(r["acc_ridge"]) - a0),
                                                    float(r["fr_dist"]) - f0, float(r["intra"]) - i0)
    return pts


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def ols(y, xs):
    X = np.column_stack([np.ones(len(y))] + list(xs))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    cov = (res ** 2).sum() / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, se, float(r2)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx])
        adj[idx] = min(run, 1.0)
    return adj


def analyse(pts, label, lr):
    doms_all = sorted({d for d, _ in pts})
    ood = [d for d in doms_all if "域内" not in d]
    doms = ood or doms_all
    eps = sorted({e for _, e in pts})
    y, x_fr, x_in = [], [], []
    for d in doms:
        for e in eps:
            vals = list(pts[(d, e)].values())
            if len(vals) >= 2:
                y.append(np.mean([v[0] for v in vals]))
                x_fr.append(np.mean([v[1] for v in vals]))
                x_in.append(np.mean([v[2] for v in vals]))
    n = len(y)
    b, se, r2 = ols(z(y), [z(x_fr)])
    slope, sse = float(b[1]), float(se[1])
    crit = float(stats.t.ppf(0.975, max(n - 2, 1)))
    ci = [round(slope - crit * sse, 3), round(slope + crit * sse, 3)]
    p1 = 2 * float(stats.t.sf(abs(slope / sse), max(n - 2, 1)))
    b_raw, se_raw, _ = ols(np.asarray(y), [np.asarray(x_fr)])
    b2, _, r2_two = ols(z(y), [z(x_fr), z(x_in)])
    per, praw = {}, []
    for d in doms_all:
        yy = [np.mean([v[0] for v in pts[(d, e)].values()]) for e in eps if len(pts[(d, e)]) >= 2]
        xx = [np.mean([v[1] for v in pts[(d, e)].values()]) for e in eps if len(pts[(d, e)]) >= 2]
        if len(yy) < 3:
            continue
        bb, ss, rr = ols(z(yy), [z(xx)])
        t = float(bb[1] / ss[1])
        pv = 2 * float(stats.t.sf(abs(t), max(len(yy) - 2, 1)))
        per[DOM_ZH.get(d.split("(")[0], d)] = {"slope": round(float(bb[1]), 3), "p": pv,
                                              "n": len(yy), "r2": round(rr, 3),
                                              "role": "in" if "域内" in d else "out"}
        if "域内" not in d:
            praw.append(pv)
    for i, d in enumerate([k for k, v in per.items() if v["role"] == "out"]):
        per[d]["p_holm"] = round(float(holm(praw)[i]), 4)
    return {"label": label, "lr": lr, "n_points": n, "n_domains": len(doms),
            "slope_std": round(slope, 3), "slope_std_ci95": ci, "p": round(p1, 6),
            "slope_raw": round(float(b_raw[1]), 5),
            "slope_raw_ci95": [round(float(b_raw[1] - crit * se_raw[1]), 5),
                               round(float(b_raw[1] + crit * se_raw[1]), 5)],
            "r2": round(r2, 3), "r2_with_intra": round(r2_two, 3),
            "dR2_intra": round(r2_two - r2, 3), "per_domain": per}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"case": CASE, "size": SIZE, "n_domains": 11, "source_domain": "s1",
           "configs": [], "split": {"train": 10493, "val": 2248, "test": 2248}}
    print("=== CORe50 外部验证(源域 s1；域外 s3/s7/s10；10 种子；16 轮)===")
    print("  %-24s %6s %10s %20s %9s %7s %10s" % ("配置", "n点", "标准化斜率", "95% CI", "p", "R²", "ΔR²(Intra)"))
    for label, tag, lr in CONFIGS:
        pts = load(tag)
        if not pts:
            print("  %-24s (结果文件缺失)" % label)
            continue
        r = analyse(pts, label, lr)
        out["configs"].append(r)
        print("  %-24s %6d %+10.3f   [%+.3f, %+.3f] %9.4f %7.3f %+10.3f" % (
            label, r["n_points"], r["slope_std"], r["slope_std_ci95"][0], r["slope_std_ci95"][1],
            r["p"], r["r2"], r["dR2_intra"]))
    print("\n=== 逐域斜率(Holm 校正)===")
    for r in out["configs"]:
        print("  %s" % r["label"])
        for d, v in r["per_domain"].items():
            ph = v.get("p_holm")
            print("    %-14s 斜率 %+.3f  p=%.4f  %s  R²=%.3f  n=%d" % (
                d, v["slope"], v["p"], ("p_holm=%.4f" % ph) if ph is not None else "(域内不校正)",
                v["r2"], v["n"]))
    prim = next((r for r in out["configs"] if r["lr"] == "3e-4"), None)
    if prim:
        p1_ok = prim["slope_std"] < 0
        f1_hit = prim["slope_std"] >= 0 and prim["slope_std_ci95"][0] * prim["slope_std_ci95"][1] > 0
        p2_ok = prim["dR2_intra"] >= 0.1
        pos = sum(1 for v in prim["per_domain"].values() if v["role"] == "out" and v["slope"] > 0)
        tot = sum(1 for v in prim["per_domain"].values() if v["role"] == "out")
        neg_pooled = prim["slope_std"] < 0
        out["verdict"] = {"P1_slope_negative": bool(p1_ok), "F1_falsified": bool(f1_hit),
                          "P2_dR2_ge_0.1": bool(p2_ok), "n_pos_per_domain": pos, "n_per_domain": tot}
        if p1_ok and pos <= tot // 2:
            verdict_text = ("该结果**支持**「在与正文三案例更接近的设定(10 类、域内接近饱和、域间位移较大)下，"
                            "距离被拉远伴随精度下降」这一候选解释；但这是相关性证据，不构成因果结论。")
        elif p1_ok:
            verdict_text = ("合并斜率方向为负，但逐域斜率以正号居多，说明该数据集上两种口径并不同向，"
                            "只能作为部分支持。")
        else:
            verdict_text = ("**未能复现**该关系(合并斜率方向不为负)，说明其成立条件比正文三案例的共性更窄，"
                            "4.2 的边界表述需据此收窄。")
        out["verdict_text"] = verdict_text
        print("\n=== 预注册判定(主检验 = lr=3e-4)===")
        print("  P1(斜率<0): %s | F1: %s | P2(ΔR²≥0.1): %s | 逐域为正 %d/%d" % (
            p1_ok, f1_hit, p2_ok, pos, tot))
        print("  判读:", verdict_text)
    (RES / "external_validation2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出:", RES / "external_validation2.json")


if __name__ == "__main__":
    main()
