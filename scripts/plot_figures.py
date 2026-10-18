# -*- coding: utf-8 -*-
"""按中文核心/SCI 期刊通用规范绘制论文图(中文标注)。

输入：results/analysis.json(由 analyze.py 生成的统计结果)
输出：figures/fig{1..4}_*.{png,pdf,tif}
    fig1_doseheat  剂量-反应热力图(行为域、列为轮数)
    fig2_lrdose    学习率剂量-反应(末轮精度变化 vs 学习率)
    fig3_summary   域内/域外精度变化对比(分组条形)
    fig4_gap       域外精度变化与特征距离变化的关系(散点 + 拟合)

制图规范(中文核心/SCI 常见约定)：
  * 拉丁字母与数字用 Times New Roman，中文回退黑体(font.family 列表式回退)
  * 坐标轴四边闭合、刻度朝内、开启次刻度；不使用网格线
  * 误差棒为 ±1 标准差；全部图均用彩色(色盲友好的 Okabe-Ito 配色)
    ——图 2/图 3 的颜色表示"域"(域内统一黑色作参照)，图 4 的颜色表示"案例"，
    图 1 为红-蓝发散色标；同时保留线型与标记形状编码，黑白打印亦可辨
  * 子图用 (a)(b)(c) 标注；图宽按单栏(约 8.9 cm)/双栏(约 17.5 cm)设置
  * 同时导出 600 dpi 位图(PNG/TIF)与矢量图(PDF)，便于期刊取用
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

CASE_ZH = {"pacs": "PACS", "organ": "OrganMNIST", "digits": "Digits"}
TAG_ZH = {"": "弱微调(3e-4)", "_strong": "强微调(1e-3)", "_lr1e4": "更弱微调(1e-4)"}
TAG_ORDER = ["_lr1e4", "", "_strong"]          # 由弱到强
TAG_STYLE = {"_lr1e4": ":", "": "--", "_strong": "-"}   # 线型：点线/虚线/实线
LR_OF_TAG = {"_lr1e4": 1e-4, "": 3e-4, "_strong": 1e-3}
DOM_SHORT = {"photo": "Photo", "art_painting": "Art", "cartoon": "Cartoon", "sketch": "Sketch",
             "axial": "轴位", "coronal": "冠状位", "sagittal": "矢状位",
             "mnist": "MNIST", "emnist": "EMNIST", "svhn": "SVHN"}
# 灰度方案(图 1、图 3、图 4 使用)：域身份用灰度/标记，微调强度用线型
GRAYS = ["#000000", "#4d4d4d", "#8c8c8c", "#b3b3b3", "#d9d9d9", "#f0f0f0"]
MARKERS = ["o", "s", "^", "v", "D"]
# 彩色方案(图 2 专用)：色盲友好的 Okabe-Ito 配色
#   域内(源域)统一用黑色作参照；每个目标域一种颜色 + 一种标记，同一图内不重复
COLOR_IN = "#000000"
DOM_COLOR = {"art_painting": "#D55E00", "cartoon": "#009E73", "sketch": "#CC79A7",
             "coronal": "#E69F00", "sagittal": "#56B4E9", "emnist": "#0072B2",
             "svhn": "#B58900"}
DOM_MARK = {"art_painting": "s", "cartoon": "^", "sketch": "v",
            "coronal": "D", "sagittal": "P", "emnist": "X", "svhn": "*"}
# 案例配色(图 4 用)：散点的颜色表示案例，与图 2/3 的"域配色"分属不同维度，各自图例已说明
CASE_COLOR = {"pacs": "#D55E00", "organ": "#009E73", "digits": "#0072B2"}
# 目标域统一排序(保证各图列序一致)
DOM_ORDER = ["art_painting", "cartoon", "sketch", "coronal", "sagittal", "emnist", "svhn"]


def setup_style():
    plt.rcParams.update({
        # font.family 直接给字体列表：matplotlib 才会做逐字回退
        # (拉丁字母/数字用 Times New Roman，中文缺字时回退到黑体)
        "font.family": ["Times New Roman", "SimHei"],
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.6,
        "axes.grid": False,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # 嵌入 TrueType，避免期刊排版出现 Type 3 字体
        "ps.fonttype": 42,
    })


def panel_label(ax, text, dx=0.02, dy=0.98):
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold")


def save_all(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("png", "pdf", "tif"):
        p = FIG / "{}.{}".format(stem, ext)
        if ext == "tif":
            fig.savefig(p, pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(p)
        outs.append(p)
    plt.close(fig)
    return outs

# ---------------- 黑白开关(本刊为黑白印刷：图表仅需黑白色) ----------------
BW = ("--color" not in sys.argv)          # 默认黑白；加 --color 出彩色版
HATCHES = ["", "//", "\\\\", "xx", "..", "||"]
LSTYLES = ["-", "--", "-.", ":"]
if BW:
    COLOR_IN = "#000000"
    _grays = ["#000000", "#3f3f3f", "#767676", "#a8a8a8", "#cfcfcf"]
    DOM_COLOR = {k: _grays[i % len(_grays)] for i, k in enumerate(DOM_COLOR)}
    CASE_COLOR = {k: _grays[i % len(_grays)] for i, k in enumerate(CASE_COLOR)}
    GRAYS = _grays



def dom_key(raw):
    """'art_painting(域外)' -> 'art_painting'"""
    return raw.split("(")[0]


# --------------------------------------------------------------------------- #
def pick_tag(A, c):
    """案例所选设置：有强微调则用强微调，否则用该案例唯一的设置。"""
    return "_strong" if "_strong" in A["per_case"][c] else sorted(A["per_case"][c])[0]


def fig1_dosecurve(A):
    """剂量-反应曲线(黑白，单栏宽)：每个案例一栏，横轴＝微调轮数，纵轴＝相对未微调基线的精度变化。

    本刊为黑白印刷，用标记形状与线型区分各域；**每块子图下方各配一份图例**，便于对号入座。
    """
    cases = [c for c in ["pacs", "organ", "digits"] if c in A["per_case"]]
    fig, axes = plt.subplots(len(cases), 1, figsize=(3.19, 2.05 * len(cases) + 0.35), squeeze=False)
    for ax, c in zip(axes[:, 0], cases):
        tag = pick_tag(A, c)
        rows = A["per_case"][c][tag]
        inx = [d for d, r in rows.items() if r["role"] == "in"][0]
        order = [inx] + sorted([d for d, r in rows.items() if r["role"] == "out"],
                               key=lambda d: DOM_ORDER.index(dom_key(d))
                               if dom_key(d) in DOM_ORDER else 99)
        ax._hs, ax._lb = [], []
        hs, labs = [], []
        ax._hs, ax._lb = [], []
        if c == "pacs":   # 减线：PACS 只画域内与代价最大的域外域(Art)；其余域外域见表 3 与图 4
            _keep = [d for d in order if rows[d]["role"] == "in"]
            _outs = [d for d in order if rows[d]["role"] == "out"]
            _outs.sort(key=lambda d: rows[d]["d_acc_pp_mean"][-1])
            order = _keep + _outs[:1]
        for q, d in enumerate(order):
            r = rows[d]
            eps = [e for e in r["epochs"] if e > 0]
            ys = [r["d_acc_pp_mean"][r["epochs"].index(e)] for e in eps]
            es = [r["d_acc_pp_std"][r["epochs"].index(e)] for e in eps]
            role = "域内" if r["role"] == "in" else "域外"
            raw = dom_key(d)
            nm = DOM_SHORT.get(raw, raw)
            # 域外统一用深灰档（q=0 域内；域外按顺序取 #4d4d4d/#8c8c8c/#b3b3b3），保证黑白下清晰
            col = COLOR_IN if role == "域内" else ["#4d4d4d", "#8c8c8c", "#b3b3b3"][q % 3]
            mk = MARKERS[q % len(MARKERS)]
            sty = LSTYLES[q % len(LSTYLES)]
            ax.errorbar(eps, ys, yerr=es, color=col, marker=mk, ls=sty, ms=3.6, lw=1.15,
                        elinewidth=0.7, capsize=1.6, capthick=0.7)
            hs.append(Line2D([], [], color=col, marker=mk, ls=sty, ms=3.6, lw=1.15))
            labs.append("{}({})".format(nm, role))
        ax.axhline(0, color="black", lw=0.8, ls=(0, (4, 2)))
        ax.set_xlabel("微调轮数", fontsize=7.5, labelpad=1)
        ax.set_ylabel("精度变化 / 百分点", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_title("{}({})".format(CASE_ZH.get(c, c), TAG_ZH.get(tag, tag).split("(")[0]),
                     fontsize=8.5, pad=2)
        for sp in ax.spines.values():
            sp.set_linewidth(0.9)
        ax.legend(hs, labs, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
                  fontsize=5.9, frameon=False, handlelength=1.9, columnspacing=0.9,
                  borderpad=0.15, labelspacing=0.35)
    fig.tight_layout(rect=(0, 0.005, 1, 1))
    fig.subplots_adjust(hspace=0.95)
    return save_all(fig, "fig1_dosecurve")

def rows_role(A, c, tag, dom):
    return A["per_case"][c][tag][dom]["role"]


def fig2_lrdose(A):
    """学习率剂量-反应：横轴为学习率(对数)，纵轴为末轮相对基线的精度变化(单栏宽，竖向两块)。"""
    cases = [c for c in ["pacs", "organ", "digits"] if c in A["per_case"]]
    fig, axes = plt.subplots(len(cases), 1, figsize=(3.19, 2.15 * len(cases)), squeeze=False)
    seen = {}
    for k, (ax, c) in enumerate(zip(axes[:, 0], cases)):
        tags = [t for t in TAG_ORDER if t in A["per_case"][c]]
        doms = list(A["per_case"][c][tags[0]].items())
        inx = [d for d, r in doms if r["role"] == "in"]
        order = inx + sorted([d for d, r in doms if r["role"] == "out"],
                             key=lambda d: DOM_ORDER.index(dom_key(d))
                             if dom_key(d) in DOM_ORDER else 99)
        for dom in order:
            raw = dom_key(dom)
            role = "域内" if A["per_case"][c][tags[0]][dom]["role"] == "in" else "域外"
            nm = DOM_SHORT.get(raw, raw)
            col = COLOR_IN if role == "域内" else DOM_COLOR.get(raw, GRAYS[1])
            mk = "o" if role == "域内" else DOM_MARK.get(raw, MARKERS[1])
            xs, ys, es = [], [], []
            for tg in tags:
                r = A["per_case"][c][tg].get(dom)
                if r is None:
                    continue
                ep = max(r["epochs"]); j2 = r["epochs"].index(ep)
                xs.append(tags.index(tg))            # 分类位置（分组柱用）
                ys.append(r["d_acc_pp_mean"][j2])
                es.append(r["d_acc_pp_std"][j2])
            _ls = LSTYLES[(len(seen)) % len(LSTYLES)] if BW else "-"
            _k = len(getattr(ax, "_hs", []))
            _n = max(len(order), 1)
            _w = 0.80 / _n
            _off = (_k - (_n - 1) / 2.0) * _w
            ax.bar([x + _off for x in xs], ys, width=_w * 0.90, yerr=es,
                   color=("#000000" if _k == 0 else "#d9d9d9"), edgecolor="black", linewidth=0.7,
                   hatch=[None, "//", "\\\\", "..", "xx"][min(_k, 4)],
                   error_kw=dict(ecolor="black", elinewidth=0.9, capsize=1.6, capthick=0.9))
            seen.setdefault((nm, role), (col, mk))
            ax._hs = getattr(ax, "_hs", []) + [Patch(facecolor=("#000000" if _k == 0 else "#d9d9d9"),
                                  edgecolor="black", linewidth=0.7,
                                  hatch=[None, "//", "\\\\", "..", "xx"][min(_k, 4)])]
            ax._lb = getattr(ax, "_lb", []) + ["{}({})".format(nm, role)]
        ax.axhline(0, color="black", lw=1.40, ls=(0, (4, 2)))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["1e-4", "3e-4", "1e-3"], fontsize=8)
        ax.set_xlabel("微调学习率", fontsize=8.5)
        ax.set_ylabel("末轮精度变化 / 百分点", fontsize=8)
        ax.tick_params(labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.9)
        ep = max(A["per_case"][c][tags[0]][order[0]]["epochs"])
        ax.set_title("{}({} 轮)".format(CASE_ZH.get(c, c), ep), fontsize=8.5, pad=2)
        if getattr(ax, "_hs", []):
            ax.legend(getattr(ax, "_hs", []), getattr(ax, "_lb", []), loc="upper center", bbox_to_anchor=(0.5, -0.30),
                      ncol=4, fontsize=5.9, frameon=False, handlelength=1.9,
                      columnspacing=0.9, borderpad=0.15, labelspacing=0.35)
    # 方案 A(显式·分格)：各子图按自身数据范围设定 y 轴，去掉多余留白
    _YLIMS = [(-24.0, 5.0), (-4.0, 21.0), (-19.0, 6.0)]   # PACS / OrganMNIST / Digits
    for _i, _ax in enumerate(fig.get_axes()):
        if _i < len(_YLIMS):
            _ax.set_ylim(*_YLIMS[_i])
    fig.tight_layout(rect=(0, 0.005, 1, 1))
    fig.subplots_adjust(hspace=0.95)
    return save_all(fig, "fig2_lrdose")


def fig3_summary(A):
    """域内/域外精度变化对比：彩色分组条形 + 误差棒(误差棒来自逐种子标准差)。"""
    strongest = A["strongest"]
    order = ["pacs|_strong", "organ|_strong", "digits|base"]
    keys = [k for k in order if k in strongest]
    labels = ["{}\n{}".format(CASE_ZH.get(strongest[k]["case"], strongest[k]["case"]),
                              TAG_ZH.get(strongest[k]["tag"], "微调")) for k in keys]

    series, errs_series = [], []
    for k in keys:
        v = strongest[k]
        case, tag, ep = v["case"], v["tag"], v["epoch"]
        bars = [("域内", v["in_d_acc_pp"], None)]
        errs = [0.0]
        rows = A["per_case"][case][tag]
        for raw in DOM_ORDER:
            for dom, val in v["out_d_acc_pp"].items():
                if dom_key(dom) == raw:
                    sd = 0.0
                    if dom in rows and ep in rows[dom]["epochs"]:
                        sd = rows[dom]["d_acc_pp_std"][rows[dom]["epochs"].index(ep)]
                    bars.append((DOM_SHORT.get(raw, raw), val, raw))
                    errs.append(sd)
        series.append(bars)
        errs_series.append(errs)

    ncol = max(len(b) for b in series)
    x = np.arange(len(keys))
    total_w = 0.78
    w = total_w / ncol
    fig, ax = plt.subplots(figsize=(3.19, 2.75))
    ymax = max(v for s_ in series for _, v, _ in s_)
    ymin = min(v for s_ in series for _, v, _ in s_)
    ax.set_ylim(min(-20, ymin - 4), max(21, ymax + 3))
    for j in range(ncol):
        vals, errs, colors = [], [], []
        for s_, es in zip(series, errs_series):
            if j < len(s_):
                nm, val, raw = s_[j]
                vals.append(val); errs.append(es[j])
                colors.append(COLOR_IN if raw is None else DOM_COLOR.get(raw, GRAYS[1]))
            else:
                vals.append(np.nan); errs.append(0.0); colors.append("white")
        _face = colors if j == 0 else ["#d9d9d9"] * len(colors)
        ax.bar(x + (j - (ncol - 1) / 2) * w, vals, w * 0.92, yerr=errs,
               color=_face, edgecolor="black", linewidth=0.6,
               hatch=(None if j == 0 else "//"),
               error_kw=dict(elinewidth=0.7, capsize=1.6, capthick=0.7, ecolor="black"))
    for j in range(ncol):
        for i, s_ in enumerate(series):
            if j < len(s_):
                val = s_[j][1]
                err = errs_series[i][j] if j < len(errs_series[i]) else 0.0
                off = (err + 0.5) if val >= 0 else -(err + 0.5)
                ax.text(x[i] + (j - (ncol - 1) / 2) * w, val + off, "{:+.1f}".format(val),
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("精度变化 / 百分点", fontsize=9)
    handles, seen = [], set()
    for s_ in series:
        for nm, _, raw in s_:
            key = raw if raw is not None else "__in__"
            if key in seen:
                continue
            seen.add(key)
            handles.append(Patch(facecolor=COLOR_IN if raw is None else DOM_COLOR.get(raw, GRAYS[1]),
                                 edgecolor="black", linewidth=0.5,
                                 hatch=(("\\\\" if raw is None else HATCHES[list(seen).index(key) % len(HATCHES)]) if BW else None),
                                 label=nm if raw is None else (nm + "(域外)")))
    _h2 = [Patch(facecolor="#000000", edgecolor="black", linewidth=0.6, label="域内"),
           Patch(facecolor="#d9d9d9", edgecolor="black", linewidth=0.6, hatch="//", label="域外")]
    fig.legend(handles=_h2, fontsize=7.5, ncol=2, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    return save_all(fig, "fig3_summary")


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def fig_protocol():
    """本文测量思路与方法流程图(黑白框图，单栏宽)：问题→协议→测量设计→数据分工→结论。"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    boxes = [
        ("研究问题",
         "同源域内评测看不到微调把特征表示推向源域\n带来的域外影响，该影响须单独测量", 0.125),
        ("配对跨域协议",
         "同一批类别；同一个在源域训练集上重新拟合的\n评估分类器；同一批随机种子并逐轮记录\n"
         "→ 域内增益与域外代价量纲一致、可直接相减", 0.165),
        ("测量设计",
         "自变量：微调强度(3 档学习率 × 轮数)\n"
         "响应量：域内／域外精度变化\n"
         "同步诊断：MMD 与 Fréchet 距离、类内紧致度", 0.155),
        ("数据分工",
         "本文数据集：PACS、MedMNIST 器官数据、\n多来源手写数字 → 发现规律\n"
         "验证数据集：OfficeHome、CORe50 → 外部检验", 0.150),
        ("结论",
         "剂量-反应证据；四项在五个数据集上一致的规律；\n"
         "一个探索性观察(两个外部数据集未复现)", 0.135),
    ]
    fig, ax = plt.subplots(figsize=(3.19, 4.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    gap = 0.034
    y = 0.985
    for k, (title, body, h) in enumerate(boxes):
        y -= h
        ax.add_patch(FancyBboxPatch((0.03, y), 0.94, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.018", clip_on=False,
                                    linewidth=0.95, edgecolor="black", facecolor="white"))
        ax.text(0.5, y + h * 0.78, title, ha="center", va="center", fontsize=6.5, fontweight="bold")
        ax.text(0.5, y + h * 0.36, body, ha="center", va="center", fontsize=5.9, linespacing=1.45)
        if k < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.003), (0.5, y - gap + 0.003),
                                         arrowstyle="-|>", mutation_scale=7, linewidth=0.9,
                                         color="black", clip_on=False))
        y -= gap
    fig.tight_layout(pad=0.2)
    return save_all(fig, "fig_protocol")

def fig4_gap(A):
    """域外精度变化 vs 特征距离变化：三案例分面板，各面板一条实线拟合(黑白)。"""
    cases = [c for c in ["pacs", "organ", "digits"] if c in A["per_case"]]
    fig, axes = plt.subplots(len(cases), 1, figsize=(3.19, 1.85 * len(cases) + 0.35), squeeze=False)
    for ax, c in zip(axes[:, 0], cases):
        xs, ys = [], []
        for tag, rows in A["per_case"][c].items():
            for raw, r in rows.items():
                if r["role"] != "out":
                    continue
                for e, dmmd, dacc in zip(r["epochs"], r["d_mmd_mean"], r["d_acc_pp_mean"]):
                    if e > 0:
                        xs.append(dmmd); ys.append(dacc)
        xs, ys = np.array(xs, float), np.array(ys, float)
        ax.scatter(xs, ys, s=11, marker="o", facecolors="#808080",
                   edgecolors="black", linewidths=0.4)
        if len(xs) > 2:
            sl, ic = np.polyfit(xs, ys, 1)
            xr = np.linspace(xs.min(), xs.max(), 40)
            ax.plot(xr, ic + sl * xr, color="black", lw=1.2, ls="-")
            rr = float(np.corrcoef(xs, ys)[0, 1])
            ax.text(0.97, 0.94, "r = {:+.2f}, n = {}".format(rr, len(xs)),
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor="#999999", lw=0.5))
        if c == "pacs":
            ax.set_xscale("symlog", linthresh=20)      # 仅 PACS 跨三个数量级，需对数刻度拉开密集区
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_xlabel("特征距离变化(MMD{})".format("，对称对数刻度" if c == "pacs" else ""),
                      fontsize=7.5, labelpad=1)
        ax.set_ylabel("域外精度变化 / 百分点", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_title(CASE_ZH.get(c, c), fontsize=8.5, pad=2)
        for sp in ax.spines.values():
            sp.set_linewidth(0.9)
    fig.tight_layout(rect=(0, 0.005, 1, 1))
    fig.subplots_adjust(hspace=0.75)
    return save_all(fig, "fig4_gap")

def main(analysis_path=None, verbose=True):
    setup_style()
    A = json.loads(Path(analysis_path or (RESULTS / "analysis.json")).read_text(encoding="utf-8"))
    made = []
    made += fig_protocol()
    made += fig1_dosecurve(A)
    made += fig2_lrdose(A)
    made += fig3_summary(A)
    made += fig4_gap(A)
    if verbose:
        for p in made:
            print("  figure:", p.name, "{:.0f} KB".format(p.stat().st_size / 1024))
    return made


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else None)
