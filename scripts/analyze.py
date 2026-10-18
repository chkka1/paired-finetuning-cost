# -*- coding: utf-8 -*-
"""汇总实验、做统计并出图(按 case × 微调强度(tag) 分组)。

产出：
  results/analysis_summary.md   全部关键数字
  results/analysis.json         机器可读(供 plot_figures.py 与 build_paper.py 使用)
  figures/fig1_overview.png     各案例在各域上的精度变化总览(热力图)
  figures/fig2_dose.png         剂量-反应曲线(均值 ± 标准差误差棒)
  figures/fig3_summary.png      域内/域外精度变化对比(分组条形)
  figures/fig4_gap.png          域外精度变化 vs 特征距离变化(散点 + 拟合)

制图规范集中在 plot_figures.py(中文标注、彩色配色、误差棒、坐标轴闭合、600 dpi + 矢量图)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300

CASE_ZH = {"pacs": "PACS(自然图像)", "organ": "OrganMNIST(腹部CT)",
           "digits": "Digits(手写/街景数字)"}
GRAYS = ["#000000", "#666666", "#AAAAAA", "#DDDDDD"]
MARKERS = ["o", "s", "^", "v"]
TAG_LABEL = {"": "lr=3e-4", "_strong": "lr=1e-3", "_lr1e4": "lr=1e-4"}
# 主序列：正文的全部数字都来自这一组(主干=ResNet-18，尺寸为各案例的主尺寸)
# 对照实验(ResNet-18@224 / ViT-B/16@224)由 robustness_check.py 单独分析
MAIN_BACKBONE = "resnet18"
MAIN_SIZE = {"pacs": 128, "organ": 64, "digits": 64}   # 主系列(正文三个案例)
# 注意：外部验证数据集 officehome 属独立分析(scripts/external_validation.py)，刻意不并入主系列，
#       以免改变正文的图 1–3 与摘要数字。


def load():
    frames = []
    for p in sorted(RESULTS.glob("*.csv")):
        if "_smoke" in p.name:
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["file"] = p.stem
        frames.append(df)
    if not frames:
        raise SystemExit("results/ 下没有可用结果")
    df = pd.concat(frames, ignore_index=True)
    df["tag"] = df["file"].str.extract(r"_s\d+(?P<tag>.*)$")["tag"]
    # 只保留主序列，避免架构/分辨率对照实验混入论文数字
    _n0 = len(df)
    df["_main_size"] = df["case"].map(MAIN_SIZE)
    df = df[(df["backbone"] == MAIN_BACKBONE) & (df["size"] == df["_main_size"])].drop(
        columns=["_main_size"]).copy()
    if len(df) != _n0:
        print("  跳过非主序列结果 %d 行(backbone/size 对照实验)" % (_n0 - len(df)))
    _missing = [c for c in MAIN_SIZE if c not in set(df["case"])]
    if _missing:
        raise SystemExit("主序列缺少案例 %s：请检查结果文件是否齐全" % _missing)
    for col in ("case", "backbone", "size", "tag", "domain", "role"):
        df[col] = df[col].fillna("").astype(str)
    return df


def deltas(df):
    key = ["case", "backbone", "size", "tag", "seed", "domain"]
    base = df[df.epoch == 0].set_index(key)[["acc_ridge", "acc_nc", "mmd", "fr_dist"]]
    base.columns = [c + "_0" for c in base.columns]
    m = df.join(base, on=key)
    for c in ["acc_ridge", "acc_nc", "mmd", "fr_dist"]:
        m["d_" + c] = m[c] - m[c + "_0"]
    m["d_acc_pp"] = m["d_acc_ridge"] * 100.0
    m["d_mmd_rel"] = np.where(m["mmd_0"] > 0, m["d_mmd"] / m["mmd_0"], np.nan)
    return m


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = load()
    m = deltas(df)
    cases = [c for c in ["pacs", "organ", "digits"] if c in set(m["case"])]
    out = {"cases": cases, "per_case": {}}

    agg = (m.groupby(["case", "tag", "epoch", "domain", "role"])[
        ["acc_ridge", "d_acc_pp", "mmd", "d_mmd"]].agg(["mean", "std", "count"]).reset_index())
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.values]

    for c in cases:
        out["per_case"][c] = {}
        for tag in sorted(agg[agg["case"] == c]["tag"].unique()):
            sub = agg[(agg["case"] == c) & (agg["tag"] == tag)]
            rows = {}
            for dom in sorted(sub["domain"].unique()):
                s = sub[sub["domain"] == dom].sort_values("epoch")
                rows[dom] = {
                    "role": s["role"].iloc[0],
                    "epochs": s["epoch"].tolist(),
                    "acc_mean": s["acc_ridge_mean"].round(4).tolist(),
                    "d_acc_pp_mean": s["d_acc_pp_mean"].round(2).tolist(),
                    "d_acc_pp_std": s["d_acc_pp_std"].round(2).fillna(0).tolist(),
                    "mmd_mean": s["mmd_mean"].round(2).tolist(),
                    "d_mmd_mean": s["d_mmd_mean"].round(2).tolist(),
                    "n_seeds": int(s["acc_ridge_count"].max()),
                }
            out["per_case"][c][tag] = rows

    # ---------- 域外代价 vs 特征距离 ----------
    tgt = m[(m["role"] == "out") & (m["epoch"] > 0)].dropna(subset=["d_acc_pp", "d_mmd"])
    pr, pp = stats.pearsonr(tgt["d_mmd"], tgt["d_acc_pp"])
    sr, sp = stats.spearmanr(tgt["d_mmd"], tgt["d_acc_pp"])
    slope, intercept, rval, pval, stderr = stats.linregress(tgt["d_mmd"], tgt["d_acc_pp"])
    per_case = {}
    for c in cases:
        s = tgt[tgt["case"] == c]
        if len(s) > 2:
            r, p = stats.pearsonr(s["d_mmd"], s["d_acc_pp"])
            per_case[c] = {"n": int(len(s)), "pearson_r": round(float(r), 4), "p": float(p)}
    out["gap_vs_cost"] = {
        "n": int(len(tgt)), "pearson_r": round(float(pr), 4), "pearson_p": float(pp),
        "spearman_r": round(float(sr), 4), "spearman_p": float(sp),
        "ols_slope": round(float(slope), 4), "ols_intercept": round(float(intercept), 3),
        "ols_r2": round(float(rval ** 2), 4), "ols_p": float(pval), "per_case": per_case,
    }

    # ---------- 各设置最终结果 ----------
    strongest = {}
    for c in cases:
        for tag in out["per_case"][c]:
            sub = m[(m["case"] == c) & (m["tag"] == tag)]
            if sub.empty:
                continue
            emax = int(sub["epoch"].max())
            s = sub[sub["epoch"] == emax]
            strongest["{}|{}".format(c, tag or "base")] = {
                "case": c, "tag": tag, "epoch": emax,
                "in_d_acc_pp": round(float(s[s["role"] == "in"]["d_acc_pp"].mean()), 2),
                "in_acc": round(float(s[s["role"] == "in"]["acc_ridge"].mean()), 4),
                "out_d_acc_pp": {dom: round(float(g["d_acc_pp"].mean()), 2)
                                 for dom, g in s[s["role"] == "out"].groupby("domain")},
                "out_mean_d_acc_pp": round(float(s[s["role"] == "out"]["d_acc_pp"].mean()), 2),
            }
    out["strongest"] = strongest

    (RESULTS / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    # ---------- 出图(期刊规范制图集中在 plot_figures.py)----------
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import plot_figures
    print("figures ->")
    plot_figures.main(RESULTS / "analysis.json")

    # ---------- 文本 ----------
    L = ["# 微调的域内收益与域外代价：汇总", ""]
    for c in cases:
        L.append("## {}".format(CASE_ZH.get(c, c)))
        for tag, rows in out["per_case"][c].items():
            L.append("\n### 微调强度 {}".format(TAG_LABEL.get(tag, tag or "base")))
            for dom, r in rows.items():
                L.append("- **{}**({}，n_seeds={})".format(
                    dom, "域内" if r["role"] == "in" else "域外", r["n_seeds"]))
                L.append("  - 精度 {}".format(["%.3f" % v for v in r["acc_mean"]]))
                L.append("  - Δ(pp) {}".format(["%+.2f" % v for v in r["d_acc_pp_mean"]]))
                L.append("  - MMD {}".format(["%.1f" % v for v in r["mmd_mean"]]))
    gv = out["gap_vs_cost"]
    L.append("\n## 域外代价 vs 特征距离变化")
    L.append("- n={}，Pearson r={}(p={:.2e})，Spearman r={}(p={:.2e})".format(
        gv["n"], gv["pearson_r"], gv["pearson_p"], gv["spearman_r"], gv["spearman_p"]))
    L.append("- OLS：Δacc = {:.3f} {}{:.3f}·ΔMMD，R²={}，p={:.2e}".format(
        gv["ols_intercept"], "+" if gv["ols_slope"] >= 0 else "", gv["ols_slope"],
        gv["ols_r2"], gv["ols_p"]))
    L.append("- 分案例：{}".format(gv["per_case"]))
    L.append("\n## 各设置最终结果(域内 vs 域外)")
    for v in strongest.values():
        L.append("- {} {}(epoch {})：域内 {:+.2f} pp，域外均值 {:+.2f} pp；各域外 {}".format(
            CASE_ZH.get(v["case"], v["case"]), TAG_LABEL.get(v["tag"], v["tag"] or "base"),
            v["epoch"], v["in_d_acc_pp"], v["out_mean_d_acc_pp"], v["out_d_acc_pp"]))
    (RESULTS / "analysis_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
