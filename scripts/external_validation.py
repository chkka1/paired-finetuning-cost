# -*- coding: utf-8 -*-
"""外部数据集验证(OfficeHome)：严格按冻结点 PREREGISTRATION.md 的统计口径实现。

口径(与冻结文本一致)：
  - 预测 P1：域外观测点上 Δacc 对 ΔFréchet 的标准化回归斜率 < 0；
  - 证伪 F1：斜率 ≥ 0 且 95% CI 不含 0；
  - 次要 P2：加入 ΔIntra 后 R² 相对单变量模型上升 ≥ 0.1；
  - 统计：**先按(域 × 轮数)对 5 个种子取均值**，再回归(避免同种子跨轮次观测点非独立)；报告效应量与 95% CI；
  - 多重比较：对 3 个域外域的斜率检验做 **Holm–Bonferroni** 校正，校正前后都报。

输入：results/officehome_resnet18_s128{,_strong,_lr1e4}.csv
输出：results/external_validation.json + 终端表
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
CASE = "officehome"
SIZE = 128
CONFIGS = [  # (标签, 文件后缀, 学习率, 轮数) —— 3 档学习率 × 2 个训练预算，全部报告
    ("lr=3e-4(16 轮，预注册主检验)", "", "3e-4", 16),
    ("lr=1e-3(16 轮)", "_strong", "1e-3", 16),
    ("lr=1e-4(16 轮)", "_lr1e4", "1e-4", 16),
    ("lr=3e-4(40 轮，附加条件)", "_e40", "3e-4", 40),
    ("lr=1e-3(40 轮)", "_strong_e40", "1e-3", 40),
    ("lr=1e-4(40 轮)", "_lr1e4_e40", "1e-4", 40),
]
DOM_ZH = {"art": "Art", "clipart": "Clipart", "product": "Product", "real world": "Real World"}


def load(tag: str):
    p = RES / ("%s_resnet18_s%d%s.csv" % (CASE, SIZE, tag))
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    base = {}
    for r in rows:
        if int(r["epoch"]) == 0:
            base[(int(r["seed"]), r["domain"])] = (float(r["acc_ridge"]), float(r["fr_dist"]),
                                                   float(r["intra"]))
    pts = {}                                    # (domain, epoch) -> {seed: (Δacc, ΔF, ΔIntra)}
    for r in rows:
        ep = int(r["epoch"])
        if ep == 0:
            continue
        k = (int(r["seed"]), r["domain"])
        if k not in base:
            continue
        a0, f0, i0 = base[k]
        pts.setdefault((r["domain"], ep), {})[int(r["seed"])] = (
            100.0 * (float(r["acc_ridge"]) - a0), float(r["fr_dist"]) - f0, float(r["intra"]) - i0)
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
    """Holm–Bonferroni：返回校正后 p 值(保持原顺序)。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def analyse(pts, label, lr):
    # 按(域 × 轮数)对种子取均值；**P1 只使用域外域**(预注册文本)，域内单独作描述性报告
    doms_all = sorted({d for d, _ in pts})
    ood = [d for d in doms_all if "域内" not in d]
    doms = ood if ood else doms_all
    eps = sorted({e for _, e in pts})
    y, x_fr, x_in = [], [], []
    for d in doms:
        for e in eps:
            vals = list(pts[(d, e)].values())
            if len(vals) < 2:
                continue
            y.append(np.mean([v[0] for v in vals]))
            x_fr.append(np.mean([v[1] for v in vals]))
            x_in.append(np.mean([v[2] for v in vals]))
    n = len(y)
    b, se, r2 = ols(z(y), [z(x_fr)])
    slope, slope_se, slope_t = float(b[1]), float(se[1]), float(b[1] / se[1])
    crit = float(stats.t.ppf(0.975, max(n - 2, 1)))
    ci = [round(slope - crit * slope_se, 3), round(slope + crit * slope_se, 3)]
    p1 = 2 * float(stats.t.sf(abs(slope_t), max(n - 2, 1)))
    # 未标准化斜率(百分点 / 原始 ΔFréchet 单位)
    b_raw, se_raw, _ = ols(np.asarray(y), [np.asarray(x_fr)])
    # P2：加入 ΔIntra
    b2, se2, r2_two = ols(z(y), [z(x_fr), z(x_in)])
    # 逐域斜率
    per_dom, praw = {}, []
    for d in doms_all:                      # 域内也报，便于对照；但只有域外进入 Holm 检验
        yy = [np.mean([v[0] for v in pts[(d, e)].values()]) for e in eps if len(pts[(d, e)]) >= 2]
        xx = [np.mean([v[1] for v in pts[(d, e)].values()]) for e in eps if len(pts[(d, e)]) >= 2]
        if len(yy) < 3:
            continue
        bb, ss, rr = ols(z(yy), [z(xx)])
        t = float(bb[1] / ss[1])
        pv = 2 * float(stats.t.sf(abs(t), max(len(yy) - 2, 1)))
        per_dom[DOM_ZH.get(d, d)] = {"slope": round(float(bb[1]), 3), "p": pv, "n": len(yy),
                                     "r2": round(rr, 3), "role": "in" if "域内" in d else "out",
                                     "in_holm": "域内不参与校正" if "域内" in d else "参与"}
        if "域内" not in d:
            praw.append(pv)
    for i, d in enumerate([k for k, v in per_dom.items() if v["role"] == "out"]):
        per_dom[d]["p_holm"] = round(float(holm(praw)[i]), 4)
    return {"label": label, "lr": lr, "n_points": n, "n_domains": len(doms),
            "slope_std": round(slope, 3), "slope_std_ci95": ci, "p": round(p1, 6),
            "slope_raw": round(float(b_raw[1]), 5), "slope_raw_ci95": [
                round(float(b_raw[1] - crit * se_raw[1]), 5), round(float(b_raw[1] + crit * se_raw[1]), 5)],
            "r2": round(r2, 3), "r2_with_intra": round(r2_two, 3),
            "dR2_intra": round(r2_two - r2, 3),
            "intra_coef": round(float(b2[2]), 3),
            "per_domain": per_dom}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"case": CASE, "size": SIZE, "configs": []}
    print("=== 外部验证：OfficeHome(源域 Real World，3 档学习率 × 2 个训练预算)===")
    print("  口径：按(域 × 轮数)对种子取均值后回归；**P1 只用域外域**；斜率 = Δacc 对 ΔFréchet 的标准化系数\n")
    print("  %-22s %8s %8s %18s %9s %7s %9s" % ("配置", "n点", "标准化斜率", "95% CI", "p", "R²", "ΔR²(Intra)"))
    for label, tag, lr, budget in CONFIGS:
        pts = load(tag)
        if not pts:
            print("  %-22s (结果文件缺失)" % label)
            continue
        r = analyse(pts, label, lr)
        r["budget"] = budget
        out["configs"].append(r)
        print("  %-22s %8d %+8.3f   [%+.3f, %+.3f] %9.4f %7.3f %+9.3f" % (
            label, r["n_points"], r["slope_std"], r["slope_std_ci95"][0], r["slope_std_ci95"][1],
            r["p"], r["r2"], r["dR2_intra"]))
    print("\n=== 逐域斜率(Holm 校正)===")
    for r in out["configs"]:
        print("  %s" % r["label"])
        for d, v in r["per_domain"].items():
            ph = v.get("p_holm")
            print("    %-12s 斜率 %+.3f  p=%.4f  %-14s R²=%.3f  n=%d" % (
                d, v["slope"], v["p"],
                ("p_holm=%.4f" % ph) if ph is not None else "(域内不校正)", v["r2"], v["n"]))
    # 预注册判定(主检验 = lr=3e-4 档)
    prim = next((r for r in out["configs"] if r["lr"] == "3e-4" and r.get("budget") == 16), None)
    # 预算稳健性：同一学习率下 16 轮与 40 轮的斜率是否都接近 0 且 CI 都跨 0
    robust = {}
    for lr in ("3e-4", "1e-3", "1e-4"):
        a16 = next((r for r in out["configs"] if r["lr"] == lr and r.get("budget") == 16), None)
        a40 = next((r for r in out["configs"] if r["lr"] == lr and r.get("budget") == 40), None)
        if a16 and a40:
            span0 = (a16["slope_std_ci95"][0] <= 0 <= a16["slope_std_ci95"][1]) and                     (a40["slope_std_ci95"][0] <= 0 <= a40["slope_std_ci95"][1])
            near0 = abs(a16["slope_std"]) < 0.2 and abs(a40["slope_std"]) < 0.2
            robust[lr] = {"slope_16": a16["slope_std"], "slope_40": a40["slope_std"],
                          "both_ci_span_zero": bool(span0), "both_near_zero": bool(near0),
                          "consistent": bool(span0 and near0)}
    out["budget_robustness"] = robust
    verdict = {}
    if prim:
        p1_ok = prim["slope_std"] < 0
        f1_hit = prim["slope_std"] >= 0 and prim["slope_std_ci95"][0] * prim["slope_std_ci95"][1] > 0
        p2_ok = prim["dR2_intra"] >= 0.1
        verdict = {"primary_lr": "3e-4", "primary_budget": 16, "P1_slope_negative": bool(p1_ok),
                   "F1_falsified": bool(f1_hit), "P2_dR2_ge_0.1": bool(p2_ok),
                   "note": "三档学习率全部报告；主检验按冻结点文本取 lr=3e-4(与正文主实验档位一致)，"
                           "其余两档作为次要结果一并呈现(不选择性报告)。"}
    out["verdict"] = verdict
    if verdict:
        print("\n=== 预注册判定(主检验 = lr=3e-4)===")
        print("  P1(斜率<0): %s | F1(斜率≥0 且 CI 不含 0): %s | P2(ΔR²≥0.1): %s" % (
            verdict["P1_slope_negative"], verdict["F1_falsified"], verdict["P2_dR2_ge_0.1"]))
    (RES / "external_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出:", RES / "external_validation.json")


if __name__ == "__main__":
    main()
