# -*- coding: utf-8 -*-
"""把 4.5 的案例内模型推进为跨案例可预测模型。

三件事：
  (1) 无量纲剂量 ρ = ΔF²/F²(0)，在各案例内拟合 Δacc = g − κ·ρ，比较 κ 是否跨案例一致；
  (2) 合并拟合(案例截距 + 共同斜率)与分案例拟合的对比；
  (3) 留一案例外推(LOCO)：用两个案例拟合、预测第三个案例，报告 R²/RMSE 并与“预测 0”基线比较；
      另做前瞻版：仅用微调前的初始距离 F²(0)(按案例内中位数无量纲化)预测最终代价。

输入：results/*.csv   输出：results/cross_case.json + 终端摘要
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TAG_FROM_NAME = re.compile(r"_s\d+(?P<tag>.*)$")
CASES = ("pacs", "organ", "digits")


MAIN_SIZE = {"pacs": 128, "organ": 64, "digits": 64}   # 主序列尺寸；对照实验(224/ViT)另行分析


def is_main(row) -> bool:
    """只保留主序列(ResNet-18 + 各案例主尺寸)，避免对照实验混入论文数字。"""
    return row.get("backbone") == "resnet18" and int(row.get("size", -1)) == MAIN_SIZE.get(row["case"], -1)


def load():
    rows = []
    for p in sorted(RESULTS.glob("*.csv")):
        if "_smoke" in p.name:
            continue
        tag = TAG_FROM_NAME.search(p.stem).group("tag")
        case = p.name.split("_")[0]
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["case"], r["tag"] = case, tag
                if not is_main(r):
                    continue
                rows.append(r)
    base = {}
    for r in rows:
        if int(r["epoch"]) == 0:
            base[(r["case"], r["tag"], r["seed"], r["domain"])] = {
                "acc": float(r["acc_ridge"]), "fr": float(r["fr_dist"]), "mmd": float(r["mmd"])}
    pts = []
    for r in rows:
        k = (r["case"], r["tag"], r["seed"], r["domain"])
        if k not in base or int(r["epoch"]) == 0 or "域内" in r["domain"]:
            continue
        f0 = base[k]["fr"]
        if f0 <= 0:
            continue
        pts.append({"case": r["case"], "tag": r["tag"], "domain": r["domain"], "seed": r["seed"],
                    "epoch": int(r["epoch"]),
                    "d_acc": 100.0 * (float(r["acc_ridge"]) - base[k]["acc"]),
                    "rho": (float(r["fr_dist"]) - f0) / f0,          # 无量纲剂量
                    "f0_rel": f0,                                     # 初始距离(前瞻用)
                    "d_mmd_rel": (float(r["mmd"]) - base[k]["mmd"]) / max(base[k]["mmd"], 1e-9)})
    # 每个案例内用初始距离的中位数做无量纲化(前瞻版)
    for c in CASES:
        med = np.median([p["f0_rel"] for p in pts if p["case"] == c]) if any(p["case"] == c for p in pts) else 1.0
        for p in pts:
            if p["case"] == c:
                p["f0_scaled"] = med and p["f0_rel"] / med or np.nan
    return pts


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = ((y - yh) ** 2).sum() / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, se, float(r2), yh


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pts = load()
    out = {"n": len(pts)}

    # ---- (1) 分案例 κ(无量纲剂量)----
    out["per_case_kappa"] = {}
    print("=== (1) 无量纲剂量的斜率 κ(单位：百分点 / 单位相对距离变化)===")
    for c in CASES:
        sub = [p for p in pts if p["case"] == c]
        y = np.array([p["d_acc"] for p in sub]); x = np.array([p["rho"] for p in sub])
        b, se, r2, _ = ols(y, [x])
        out["per_case_kappa"][c] = {"kappa": round(float(b[1]), 2), "se": round(float(se[1]), 2),
                                    "ci95": [round(float(b[1] - 1.96 * se[1]), 2),
                                             round(float(b[1] + 1.96 * se[1]), 2)],
                                    "intercept": round(float(b[0]), 2), "r2": round(r2, 3), "n": len(sub)}
        print("  %-7s κ=%+7.2f (se %5.2f, CI [%+.2f, %+.2f])  截距 g=%+6.2f  R²=%.3f  n=%d" % (
            c, b[1], se[1], b[1] - 1.96 * se[1], b[1] + 1.96 * se[1], b[0], r2, len(sub)))

    # ---- (2) 合并：共同斜率 + 案例截距 ----
    print("\n=== (2) 合并拟合(案例截距 + 共同斜率)===")
    y = np.array([p["d_acc"] for p in pts]); x = np.array([p["rho"] for p in pts])
    dummies = [np.array([1.0 if p["case"] == c else 0.0 for p in pts]) for c in CASES[1:]]
    b, se, r2, _ = ols(y, [x] + dummies)
    out["pooled_common_slope"] = {"kappa": round(float(b[1]), 2), "se": round(float(se[1]), 2),
                                  "r2": round(r2, 3), "n": len(pts)}
    print("  共同 κ=%+.2f (se %.2f)  R²=%.3f  n=%d" % (b[1], se[1], r2, len(pts)))
    # 与“每案例各自斜率”比较(F 检验)
    sse_full = sum(
        ((np.array([p["d_acc"] for p in pts if p["case"] == c])
          - ols(np.array([p["d_acc"] for p in pts if p["case"] == c]),
                [np.array([p["rho"] for p in pts if p["case"] == c])])[3]) ** 2).sum() for c in CASES)
    sse_red = ((y - ols(y, [x] + dummies)[3]) ** 2).sum()
    df1, df2 = 2 * (len(CASES) - 1), len(pts) - 3 - 1
    F = ((sse_red - sse_full) / df1) / (sse_full / df2)
    out["common_slope_F"] = {"F": round(float(F), 2), "df": [df1, df2]}
    print("  共同斜率假设的 F 检验：F(%d,%d)=%.2f(>%.2f 时拒绝“κ 跨案例相同”)" % (
        df1, df2, F, 2.60))

    # ---- (3) 留一案例外推 ----
    print("\n=== (3) 留一案例外推(拟合 2 案例 → 预测第 3 案例)===")
    lo = {}
    for held in CASES:
        tr = [p for p in pts if p["case"] != held]
        te = [p for p in pts if p["case"] == held]
        ytr = np.array([p["d_acc"] for p in tr]); xtr = np.array([p["rho"] for p in tr])
        dum = [np.array([1.0 if p["case"] == c else 0.0 for p in tr]) for c in CASES if c != held][1:]
        b, se, r2, _ = ols(ytr, [xtr] + dum)
        yte = np.array([p["d_acc"] for p in te]); xte = np.array([p["rho"] for p in te])
        pred = b[0] + b[1] * xte          # 用被预测案例的截距≈训练截距(保守)
        rmse = float(np.sqrt(((yte - pred) ** 2).mean()))
        rmse0 = float(np.sqrt((yte ** 2).mean()))
        r2te = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
        lo[held] = {"kappa_train": round(float(b[1]), 2), "rmse": round(rmse, 2),
                    "rmse_zero_baseline": round(rmse0, 2), "r2_test": round(float(r2te), 3), "n": len(te)}
        print("  预测 %-7s：RMSE=%.2f pp(基线“预测 0”=%.2f)R²_test=%.3f n=%d" % (
            held, rmse, rmse0, r2te, len(te)))
    out["loco"] = lo
    out["loco_mean"] = {"rmse": round(float(np.mean([v["rmse"] for v in lo.values()])), 2),
                        "rmse_zero": round(float(np.mean([v["rmse_zero_baseline"] for v in lo.values()])), 2),
                        "r2": round(float(np.mean([v["r2_test"] for v in lo.values()])), 3)}

    # ---- (4) 前瞻版：只用微调前的初始距离预测最终代价 ----
    print("\n=== (4) 前瞻版：仅用微调前初始距离预测最终域外代价 ===")
    fin = {}
    for c in CASES:
        sub = {p["tag"] + "|" + p["domain"] + "|" + p["seed"]: p for p in pts if p["case"] == c}
        ep = {}
        for p in pts:
            if p["case"] != c:
                continue
            key = p["tag"] + "|" + p["domain"] + "|" + p["seed"]
            ep[key] = max(ep.get(key, 0), p["epoch"])
        rows = [(pts_i["f0_scaled"], pts_i["d_acc"]) for k, pts_i in sub.items()
                if pts_i["epoch"] == ep.get(k)]
        fin[c] = np.array(rows)
        if len(rows) > 2:
            xs = fin[c][:, 0]; ys = fin[c][:, 1]
            r = float(np.corrcoef(xs, ys)[0, 1])
            print("  %-7s 初始距离(案例内中位数=1)与最终 Δacc 的相关 r=%+.3f(n=%d)" % (c, r, len(rows)))
            out.setdefault("prospective", {})[c] = {"r": round(r, 3), "n": len(rows)}
    allpts = np.vstack([fin[c] for c in CASES if len(fin[c]) > 2])
    r_all = float(np.corrcoef(allpts[:, 0], allpts[:, 1])[0, 1])
    out["prospective"]["pooled_r"] = round(r_all, 3)
    print("  合并三案例：r=%+.3f(n=%d)" % (r_all, len(allpts)))

    (RESULTS / "cross_case.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出:", RESULTS / "cross_case.json")


if __name__ == "__main__":
    main()
