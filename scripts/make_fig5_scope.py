# -*- coding: utf-8 -*-
"""图 5：4.6 节模型适用范围的三个画面(单栏宽 8.1 cm)。

(a) Δacc 对无量纲剂量 ρ = ΔF²/F²(0)：三案例各自的拟合线——符号一致(全为负)、斜率不同；
(b) 留一案例外推的 RMSE 与“一律预测 0”基线：外推并不优于基线；
(c) 微调前初始距离 F²(0)(案例内中位数=1)对最终 Δacc：相关为正——初始距离只决定幅度，不决定符号。

输入：results/*.csv、results/cross_case.json
输出：figures/fig5_scope.{png,pdf,tif}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_case_model as CCM  # noqa: E402
from plot_figures import CASE_COLOR, FIG, save_all, setup_style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ZH = {"pacs": "PACS", "organ": "OrganMNIST", "digits": "Digits"}


def ols(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ b
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, float(r2)


def main():
    setup_style()
    pts = CCM.load()
    cc = json.loads((RESULTS / "cross_case.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, 1, figsize=(3.19, 5.9), squeeze=False)

    # ---------- (a) Δacc ~ ρ ----------
    ax = axes[0, 0]
    handles = []
    for c in ("pacs", "organ", "digits"):
        sub = [p for p in pts if p["case"] == c]
        x = np.array([p["rho"] for p in sub]); y = np.array([p["d_acc"] for p in sub])
        col = CASE_COLOR[c]
        ax.scatter(x, y, s=5, facecolors=col, edgecolors="none", alpha=0.45)
        b, r2 = ols(x, y)
        xr = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 40)
        ax.plot(xr, b[0] + b[1] * xr, color=col, lw=1.5)
        handles.append(Line2D([], [], color=col, lw=1.5, marker="o", ms=3,
                              label="{} κ={:.2f} (R²={:.2f})".format(ZH[c], b[1], r2)))
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-28, 26)
    ax.set_xlabel("无量纲剂量 ΔF²/F²(0)", fontsize=8)
    ax.set_ylabel("Δacc / 百分点", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(handles=handles, fontsize=6.2, loc="lower left", frameon=False,
              handlelength=1.4, labelspacing=0.25)
    ax.text(0.985, 0.05, "符号一致为负，斜率不相同\n共同斜率假设被拒绝(F=5.87)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#AAAAAA", lw=0.5))
    ax.text(0.015, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, fontweight="bold")

    # ---------- (b) 留一外推 vs 基线 ----------
    ax = axes[1, 0]
    xs = np.arange(3)
    w = 0.36
    lo = [cc["loco"][c]["rmse"] for c in ("pacs", "organ", "digits")]
    bl = [cc["loco"][c]["rmse_zero_baseline"] for c in ("pacs", "organ", "digits")]
    ax.bar(xs - w / 2, lo, w * 0.92, color=[CASE_COLOR[c] for c in ("pacs", "organ", "digits")],
           edgecolor="black", linewidth=0.5, label="留一外推 RMSE")
    ax.bar(xs + w / 2, bl, w * 0.92, color="white", edgecolor="black", linewidth=0.5,
           hatch="//", label="基线(预测 0)")
    for i, (v1, v2) in enumerate(zip(lo, bl)):
        ax.text(i - w / 2, v1 + 0.4, "{:.1f}".format(v1), ha="center", va="bottom", fontsize=6.2)
        ax.text(i + w / 2, v2 + 0.4, "{:.1f}".format(v2), ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([ZH[c] for c in ("pacs", "organ", "digits")], fontsize=7.5)
    ax.set_ylabel("RMSE / 百分点", fontsize=8)
    ax.set_ylim(0, max(lo + bl) * 1.28)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.2, loc="upper right", frameon=False, ncol=1, handlelength=1.4,
              borderpad=0.2, labelspacing=0.25)
    ax.text(0.015, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, fontweight="bold")

    # ---------- (c) 初始距离 ~ 最终代价 ----------
    ax = axes[2, 0]
    handles = []
    for c in ("pacs", "organ", "digits"):
        sub = {p["tag"] + "|" + p["domain"] + "|" + p["seed"]: p for p in pts if p["case"] == c}
        last = {}
        for p in pts:
            if p["case"] != c:
                continue
            k = p["tag"] + "|" + p["domain"] + "|" + p["seed"]
            if k not in last or p["epoch"] > last[k]["epoch"]:
                last[k] = p
        x = np.array([last[k]["f0_scaled"] for k in sub if k in last])
        y = np.array([last[k]["d_acc"] for k in sub if k in last])
        col = CASE_COLOR[c]
        ax.scatter(x, y, s=11, facecolors=col, edgecolors="black", linewidths=0.4)
        b, _ = ols(x, y)
        xr = np.linspace(x.min(), x.max(), 20)
        ax.plot(xr, b[0] + b[1] * xr, color=col, lw=1.2)
        r = float(np.corrcoef(x, y)[0, 1])
        handles.append(Line2D([], [], color=col, lw=1.2, marker="o", ms=3,
                              label="{} r={:+.2f}".format(ZH[c], r)))
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xlabel("微调前初始距离(案例内中位数 = 1)", fontsize=8)
    ax.set_ylabel("最终 Δacc / 百分点", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(handles=handles, fontsize=6.2, loc="lower right", frameon=False, handlelength=1.4)
    ax.text(0.015, 0.97, "(c)", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, fontweight="bold")

    fig.tight_layout(h_pad=1.1)
    for ext in save_all(fig, "fig5_scope"):
        print("  figure:", ext.name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
