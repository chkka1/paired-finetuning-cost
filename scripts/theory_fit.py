# -*- coding: utf-8 -*-
"""距离-代价关系的可检验模型拟合(用于正文表 5)。

模型(对域外、逐轮、逐种子的观测点，变量先标准化)：
    Δacc = g_d − κ · ΔF²(S, d) + γ · ΔIntra + ε
其中 Δ 均相对第 0 轮(未微调)基线；F² 用 Fréchet 距离，ΔIntra 为类内紧致度变化。
理论依据见正文 4.5：Ben-David 界 → MMD/Fréchet → 间隔式 → 一阶线性化。

输出：results/theory_fit.json
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
METRICS = ("acc_ridge", "acc_nc", "mmd", "fr_dist", "intra")


MAIN_SIZE = {"pacs": 128, "organ": 64, "digits": 64}   # 主序列尺寸；对照实验(224/ViT)另行分析


def is_main(row) -> bool:
    """只保留主序列(ResNet-18 + 各案例主尺寸)，避免对照实验混入论文数字。"""
    return row.get("backbone") == "resnet18" and int(row.get("size", -1)) == MAIN_SIZE.get(row["case"], -1)


def load_points():
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
            base[(r["case"], r["tag"], r["seed"], r["domain"])] = {m: float(r[m]) for m in METRICS}
    pts = []
    for r in rows:
        k = (r["case"], r["tag"], r["seed"], r["domain"])
        if k not in base or int(r["epoch"]) == 0 or "域内" in r["domain"]:
            continue
        pts.append({
            "case": r["case"],
            "d_acc": 100.0 * (float(r["acc_ridge"]) - base[k]["acc_ridge"]),
            "d_acc_nc": 100.0 * (float(r["acc_nc"]) - base[k]["acc_nc"]),
            "d_mmd": float(r["mmd"]) - base[k]["mmd"],
            "d_fr": float(r["fr_dist"]) - base[k]["fr_dist"],
            "d_intra": float(r["intra"]) - base[k]["intra"],
        })
    return pts


def z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-12)


def ols(y, cols):
    """cols: list of 1-D arrays(已标准化)。返回系数、标准误、R²。"""
    X = np.column_stack([np.ones(len(y))] + list(cols))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = (resid ** 2).sum() / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, se, float(r2)


def fit_subset(sub):
    y = z([p["d_acc"] for p in sub])
    out = {"n": len(sub)}
    for lab, key in (("mmd", "d_mmd"), ("fr", "d_fr")):
        b, se, r2 = ols(y, [z([p[key] for p in sub])])
        out[lab] = {"slope": round(float(b[1]), 3), "se": round(float(se[1]), 3),
                    "t": round(float(b[1] / se[1]), 1), "r2": round(r2, 3)}
    b, se, r2 = ols(y, [z([p["d_fr"] for p in sub]), z([p["d_intra"] for p in sub])])
    out["two_var"] = {"fr": round(float(b[1]), 3), "fr_se": round(float(se[1]), 3),
                      "intra": round(float(b[2]), 3), "intra_se": round(float(se[2]), 3),
                      "r2": round(r2, 3)}
    out["acc_nc_corr"] = round(float(np.corrcoef([p["d_acc"] for p in sub],
                                                [p["d_acc_nc"] for p in sub])[0, 1]), 3)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pts = load_points()
    out = {"n": len(pts), "per_case": {}}
    for c in ("pacs", "organ", "digits"):
        sub = [p for p in pts if p["case"] == c]
        if len(sub) > 3:
            out["per_case"][c] = fit_subset(sub)
    # 合并：未剔除案例效应 vs 案例内标准化
    y_raw = z([p["d_acc"] for p in pts]); x_raw = z([p["d_fr"] for p in pts])
    b1, se1, r1 = ols(y_raw, [x_raw])
    zx, zy = [], []
    for c in ("pacs", "organ", "digits"):
        sub = [p for p in pts if p["case"] == c]
        if len(sub) > 3:
            zx += list(z([p["d_fr"] for p in sub])); zy += list(z([p["d_acc"] for p in sub]))
    b2, se2, r2 = ols(np.array(zy), [np.array(zx)])
    out["pooled"] = {"raw_slope": round(float(b1[1]), 3), "raw_r2": round(r1, 3),
                     "std_slope": round(float(b2[1]), 3), "std_se": round(float(se2[1]), 3),
                     "std_t": round(float(b2[1] / se2[1]), 1), "std_r2": round(r2, 3),
                     "n": len(zx)}
    (RESULTS / "theory_fit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 表 5 数据 ===")
    for c, v in out["per_case"].items():
        t = v["two_var"]
        print("  %-7s ΔFréchet=%+.3f(se %.3f) ΔIntra=%+.3f(se %.3f) R²=%.3f n=%d | 单变量 ΔMMD %+.3f ΔFréchet %+.3f | corr(acc_ridge,acc_nc)=%+.3f"
              % (c, t["fr"], t["fr_se"], t["intra"], t["intra_se"], t["r2"], v["n"],
                 v["mmd"]["slope"], v["fr"]["slope"], v["acc_nc_corr"]))
    p = out["pooled"]
    print("  合并 未标准化 斜率=%+.3f R²=%.3f | 案例内标准化 斜率=%+.3f(t=%.1f) R²=%.3f n=%d"
          % (p["raw_slope"], p["raw_r2"], p["std_slope"], p["std_t"], p["std_r2"], p["n"]))
    print("已写出:", RESULTS / "theory_fit.json")


if __name__ == "__main__":
    main()
