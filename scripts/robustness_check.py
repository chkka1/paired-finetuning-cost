# -*- coding: utf-8 -*-
"""C4 稳健性检验(用于正文 4.7 与表 7)：把“主干 / 分辨率 / 批大小 / 学习率”四者的效应分开。

五个配置统一取 16 轮、种子 0~2，并在同一轮数上比较：
    ResNet-18@128 (lr=3e-4, b=64)   ← 主实验
    ResNet-18@224 (lr=3e-4, b=32)
    ResNet-18@224 (lr=3e-4, b=16)   ← 补跑，与 ViT 的批大小对齐
    ViT-B/16@224  (lr=3e-4, b=16)
    ViT-B/16@224  (lr=1e-4, b=32)   ← 补跑，隔离学习率

输出：results/robustness.json(含各配置的域内/域外变化、距离-代价斜率与其 95% CI，
以及“批大小效应 / 主干效应 / 学习率效应”三组对照差值)。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EPOCH = 16
SEEDS = (0, 1, 2)
OUT_DOMAINS = ("art_painting", "cartoon", "sketch")

# 标签, 结果文件, (主干, 尺寸cm, 学习率, 批大小)
CONFIGS = [
    ("ResNet-18@128", "pacs_resnet18_s128.csv", ("ResNet-18", 128, "3e-4", 64)),
    ("ResNet-18@224", "pacs_resnet18_s224_r224.csv", ("ResNet-18", 224, "3e-4", 32)),
    ("ResNet-18@224b16", "pacs_resnet18_s224_r224b16.csv", ("ResNet-18", 224, "3e-4", 16)),
    ("ResNet-18@224lr4", "pacs_resnet18_s224_r224lr4.csv", ("ResNet-18", 224, "1e-4", 32)),
    ("ViT-B/16@224b16", "pacs_vit_b_16_s224_v224.csv", ("ViT-B/16", 224, "3e-4", 16)),
    ("ViT-B/16@224lr4", "pacs_vit_b_16_s224_v224lr4.csv", ("ViT-B/16", 224, "1e-4", 32)),
]


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    base = {}
    for r in rows:
        if int(r["epoch"]) == 0:
            base[(int(r["seed"]), r["domain"])] = {"acc": float(r["acc_ridge"]),
                                                   "fr": float(r["fr_dist"])}
    out = {}
    for r in rows:
        ep = int(r["epoch"])
        if ep == 0 or ep > EPOCH or int(r["seed"]) not in SEEDS:
            continue
        k = (int(r["seed"]), r["domain"])
        if k not in base:
            continue
        out[(k[0], k[1], ep)] = {"d_acc": 100.0 * (float(r["acc_ridge"]) - base[k]["acc"]),
                                 "d_fr": float(r["fr_dist"]) - base[k]["fr"],
                                 "final": ep == EPOCH}
    return out


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def ols(y, x):
    A = np.column_stack([np.ones(len(y)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    dof = max(len(y) - 2, 1)
    se = np.sqrt((res ** 2).sum() / dof * np.linalg.inv(A.T @ A).diagonal())
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, se, float(r2)


def dom_mean(data, dom, key="d_acc"):
    """末轮、各种子平均。"""
    vals = [v[key] for (sd, d, ep), v in data.items() if dom in d and v.get("final")]
    return float(np.mean(vals)) if vals else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"epoch": EPOCH, "seeds": list(SEEDS), "configs": {}}
    print("=== 统一在弱微调第 %d 轮、种子 %s 上比较 ===" % (EPOCH, list(SEEDS)))
    print("  %-18s %5s %5s %7s %7s %7s %7s %8s %7s %6s" % (
        "配置", "lr", "batch", "域内", "Art", "Cartoon", "Sketch", "域外均值", "斜率", "R²"))
    for label, fname, meta in CONFIGS:
        data = load(fname)
        if not data:
            print("  %-18s (缺结果文件 %s)" % (label, fname))
            continue
        backbone, size, lr, batch = meta
        ins = dom_mean(data, "photo")
        outs = {d: dom_mean(data, d) for d in OUT_DOMAINS}
        outs = {k: v for k, v in outs.items() if v is not None}
        ood_mean = float(np.mean(list(outs.values()))) if outs else None

        def fit(sel):
            pts = [(v["d_fr"], v["d_acc"]) for (sd, d, ep), v in data.items()
                   if "域内" not in d and "photo" not in d and sel(v)]
            if len(pts) <= 3:
                return None, None, None, len(pts)
            y = z([q[1] for q in pts]); x = z([q[0] for q in pts])
            b, se, r2 = ols(y, x)
            return float(b[1]), float(se[1]), float(r2), len(pts)

        slope, se_slope, r2, n_all = fit(lambda v: True)
        sfin, _, r2fin, n_fin = fit(lambda v: v["final"])
        ci = None
        if slope is not None:
            crit = float(st.t.ppf(0.975, max(n_all - 2, 1)))
            ci = [round(slope - crit * se_slope, 2), round(slope + crit * se_slope, 2)]
        out["configs"][label] = {
            "backbone": backbone, "size": size, "lr": lr, "batch": batch,
            "in_domain": round(ins, 2) if ins is not None else None,
            "out_domains": {d: round(v, 2) for d, v in outs.items()},
            "out_mean": round(ood_mean, 2) if ood_mean is not None else None,
            "slope": round(slope, 3) if slope is not None else None,
            "slope_se": round(se_slope, 4) if se_slope is not None else None,
            "slope_t": round(slope / se_slope, 1) if slope is not None and se_slope else None,
            "ci95": ci, "r2": round(r2, 3) if r2 is not None else None,
            "n_out_points": n_all,
            "slope_final": round(sfin, 3) if sfin is not None else None,
            "r2_final": round(r2fin, 3) if r2fin is not None else None,
            "n_final_points": n_fin,
            "n_seeds": len({sd for sd, _, _ in data}),
        }
        c = out["configs"][label]
        print("  %-18s %5s %5d %+7.2f %+7.2f %+7.2f %+7.2f %+8.2f %+7.2f %6.2f" % (
            label, lr, batch, c["in_domain"], c["out_domains"].get("art_painting", float("nan")),
            c["out_domains"].get("cartoon", float("nan")), c["out_domains"].get("sketch", float("nan")),
            c["out_mean"], c["slope"] if c["slope"] is not None else float("nan"),
            c["r2"] if c["r2"] is not None else float("nan")))

    # ---- 效应分解：批大小 / 主干 / 学习率 ----
    def diff(la, lb, what):
        A, B = out["configs"].get(la), out["configs"].get(lb)
        if not A or not B or A["slope"] is None or B["slope"] is None:
            return None
        d_slope = B["slope"] - A["slope"]
        se_d = float(np.hypot(A["slope_se"], B["slope_se"]))
        rec = {"对比": what, "A": la, "B": lb,
               "d_out_mean": round(B["out_mean"] - A["out_mean"], 2),
               "d_in_domain": round(B["in_domain"] - A["in_domain"], 2),
               "slope_A": A["slope"], "slope_B": B["slope"],
               "d_slope": round(d_slope, 2),
               "d_slope_ci95": [round(d_slope - 1.96 * se_d, 2), round(d_slope + 1.96 * se_d, 2)]}
        rec["d_slope_sig"] = bool(rec["d_slope_ci95"][0] * rec["d_slope_ci95"][1] > 0)
        return rec

    out["effects"] = [x for x in [
        diff("ResNet-18@224", "ResNet-18@224b16", "批大小 32→16(同主干、同分辨率、同学习率)"),
        diff("ResNet-18@224b16", "ViT-B/16@224b16", "主干 ResNet-18→ViT-B/16(同分辨率、批大小、学习率)"),
        diff("ViT-B/16@224b16", "ViT-B/16@224lr4", "学习率 3e-4→1e-4(同主干；批大小 16→32)"),
    ] if x]
    print("\n=== 效应分解(域外均值变化 / 斜率变化)===")
    for e in out["effects"]:
        print("  %-52s 域外均值 %+6.2f pp | 斜率 %+.2f → %+.2f(差 %+.2f，CI [%+.2f, %+.2f]，%s)" % (
            e["对比"], e["d_out_mean"], e["slope_A"], e["slope_B"], e["d_slope"],
            e["d_slope_ci95"][0], e["d_slope_ci95"][1], "显著" if e["d_slope_sig"] else "不显著"))
    # 2×2 网格：主干 × 学习率(同分辨率、同轮数)
    def _g(backbone, lr):
        for v in out["configs"].values():
            if v["backbone"] == backbone and v["lr"] == lr and v["size"] == 224:
                return v
        return None
    grid = {}
    for bb in ("ResNet-18", "ViT-B/16"):
        for lr in ("3e-4", "1e-4"):
            v = _g(bb, lr)
            grid["{}@{}".format(bb, lr)] = None if not v else {
                "out_mean": v["out_mean"], "in_domain": v["in_domain"],
                "slope": v["slope"], "ci95": v["ci95"], "batch": v["batch"]}
    out["grid_2x2"] = grid
    print("\n=== 2×2 网格(主干 × 学习率，224×224)===")
    for bb in ("ResNet-18", "ViT-B/16"):
        for lr in ("3e-4", "1e-4"):
            g = grid["{}@{}".format(bb, lr)]
            if g:
                print("  %-10s lr=%-5s (b%-2d) 域内 %+6.2f | 域外均值 %+7.2f | 斜率 %+.2f CI %s" % (
                    bb, lr, g["batch"], g["in_domain"], g["out_mean"], g["slope"], g["ci95"]))
            else:
                print("  %-10s lr=%-5s 缺失" % (bb, lr))
    neg = [c for c, v in out["configs"].items() if v.get("slope") is not None and v["slope"] < 0]
    out["slopes_negative"] = {"n_negative": len(neg), "n_total": len(out["configs"]), "which": neg}
    print("\n  斜率为负的配置：%d/%d  %s" % (len(neg), len(out["configs"]), neg))

    (RESULTS / "robustness.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出:", RESULTS / "robustness.json")


if __name__ == "__main__":
    main()
