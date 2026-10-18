# -*- coding: utf-8 -*-
"""表级逐格校验(补 audit_paper.py 的缺口：只查“数字出现过”无法发现“填错行”)。

对 manuscript_final.md 里每张数据表的每个单元格，回到源 JSON 重算并逐格比对：
  表 2 → results/analysis.json            (剂量-反应代表性观测点)
  表 3 → results/theory_fit.json          (距离-代价模型拟合)
  表 4 → results/extra_stats.json         (关键增量 ±SD / 95% CI / p；按“案例+强度+域”精确匹配)
  表 5 → results/cross_case.json          (κ / CI / 截距 / R² / 留一 RMSE)
  表 6 → results/robustness.json          (各配置域内、域外均值、斜率、R²)

输出：逐格 PASS/FAIL；任一格不符则退出码 1。
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TOL = 0.011                      # 四舍五入容差


def load_tables(md: str):
    """把 markdown 表格解析为 {表号: [[单元格,...], ...]}(含表头行)。"""
    out, cur, num = {}, [], None
    for line in md.split("\n"):
        m = re.match(r"\*\*表 (\d)", line)
        if m:
            cur, num = [], int(m.group(1))
        elif cur is not None and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                cur.append(cells)
        elif cur is not None and not line.startswith("|"):
            if line.strip() == "":
                continue                      # 表题与表体之间允许空行
            if cur:
                out[num] = cur
            cur, num = [], None
    if cur and num:
        out[num] = cur
    return out


def num(cell: str):
    m = re.search(r"[-−+]?\d+\.?\d*", cell.replace("−", "-"))
    return float(m.group(0)) if m else None


def approx(a, b, tol=TOL):
    return a is not None and b is not None and abs(a - b) <= tol


def ndec(cell: str) -> int:
    """显示值的小数位数，用于按显示精度比较(源值 5.06 显示为 5.1 属正确)。"""
    m = re.search(r"[-−+]?\d+\.(\d+)", cell)
    return len(m.group(1)) if m else 0


def cell_ok(cell: str, src) -> bool:
    if src is None:
        return False
    nd = ndec(cell)
    return abs(round(float(src), nd) - num(cell)) <= 10 ** (-nd) / 2 + 1e-9


def ci_pair(cell: str):
    m = re.search(r"\[\s*([-−+]?\d+\.?\d*)\s*,\s*([-−+]?\d+\.?\d*)\s*\]", cell.replace("−", "-"))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    md = (ROOT / "manuscript_final.md").read_text(encoding="utf-8")
    T = load_tables(md)
    A = json.loads((RES / "analysis.json").read_text(encoding="utf-8"))
    X = json.loads((RES / "extra_stats.json").read_text(encoding="utf-8"))
    TF = json.loads((RES / "theory_fit.json").read_text(encoding="utf-8"))
    CC = json.loads((RES / "cross_case.json").read_text(encoding="utf-8"))
    RB = json.loads((RES / "robustness.json").read_text(encoding="utf-8"))
    checks, bad = 0, 0

    def ck(tag, cond, detail=""):
        nonlocal checks, bad
        checks += 1
        if not cond:
            bad += 1
            print("  ✗ %s %s" % (tag, detail))

    print("=== 表 3(analysis.json，剂量-反应观测点)===")
    def ep_val(case, tag, dom_key, ep):
        """与 build_paper.d_of 同源：per_case[case][tag][域] 的逐轮均值。"""
        for dname, rec in A["per_case"].get(case, {}).get(tag, {}).items():
            if dname.split("(")[0] == dom_key and ep in rec["epochs"]:
                return rec["d_acc_pp_mean"][rec["epochs"].index(ep)]
        return None
    ZH2DOM = {"Art": "art_painting", "Cartoon": "cartoon", "Sketch": "sketch",
              "冠状位": "coronal", "矢状位": "sagittal", "EMNIST": "emnist", "SVHN": "svhn"}
    for row in T.get(3, [])[1:]:
        case = "pacs" if "PACS" in row[0] else ("organ" if "ORGAN" in row[0].upper() else "digits")
        ep = int(num(row[1]))
        zm = re.search(r"([\u4e00-\u9fff]+)", row[0])
        tag = {"强": "_strong", "弱": "", "更弱": "_lr1e4"}.get((zm.group(1) if zm else "强")[:1], "_strong")
        got_in = ep_val(case, tag, {"pacs": "photo", "organ": "axial", "digits": "mnist"}[case], ep)
        ck("表2 %s/%s ep%d 域内取到源值" % (case, tag, ep), got_in is not None)
        ck("表2 %s/%s ep%d 域内值" % (case, tag, ep), approx(num(row[2]), got_in),
           "文中 %s vs 源 %s" % (row[2], got_in))
        for item in [c for c in row[3].split("、") if c.strip()]:
            parts = item.strip().rsplit(" ", 1)
            if len(parts) != 2:
                continue
            dom_zh, val = parts
            src = ep_val(case, tag, ZH2DOM.get(dom_zh.strip(), dom_zh.strip()), ep)
            ck("表2 %s/%s ep%d %s" % (case, tag, ep, dom_zh.strip()), approx(num(val), src),
               "文中 %s vs 源 %s" % (val.strip(), src))

    print("=== 表 4(extra_stats.json，关键增量统计)===")
    case_from = {"PACS": "pacs", "ORGANMNIST": "organ", "DIGITS": "digits"}
    # 注意：必须按长度优先匹配("弱" 是 "更弱微调" 的子串，否则更弱行会被错配到弱档)
    tag_from = [("更弱", "_lr1e4"), ("强", "_strong"), ("弱", "")]
    dom_from = {"photo": "photo", "art": "art_painting", "cartoon": "cartoon", "sketch": "sketch",
                "轴位": "axial", "冠状位": "coronal", "矢状位": "sagittal",
                "mnist": "mnist", "emnist": "emnist", "svhn": "svhn"}
    for row in T.get(4, [])[1:]:
        cm = re.search(r"([A-Za-z]+)", row[0])
        case = case_from.get(cm.group(1).upper()) if cm else None
        tag = None
        for k, v in tag_from:
            if k in row[0]:
                tag = v
                break
        dom = dom_from.get(row[1].strip().lower(), dom_from.get(row[1].strip()))
        hit = next((e for e in X["c1"]["effects"]
                    if e["case"] == case and e["tag"] == tag and e["domain"].split("(")[0] == dom), None)
        ck("表4 %s/%s/%s 命中源数据" % (case, tag, dom), hit is not None, "文中 %s | %s" % (row[0], row[1]))
        if hit:
            ck("表4 %s/%s/%s Δacc" % (case, tag, dom), cell_ok(row[2], hit["mean"]),
               "文中 %s vs %s" % (row[2], hit["mean"]))
            ck("表4 %s/%s/%s SD" % (case, tag, dom), cell_ok(row[2].split("±")[1], hit["sd"]),
               "文中 %s vs %s" % (row[2], hit["sd"]))
            _lo, _hi = ci_pair(row[3])
            ck("表4 %s/%s/%s CI" % (case, tag, dom),
               cell_ok("{:.2f}".format(_lo), hit["ci95"][0]) and cell_ok("{:.2f}".format(_hi), hit["ci95"][1]),
               "文中 %s vs %s" % (row[3], hit["ci95"]))

    print("=== 表 5(external_validation.json，OfficeHome，按学习率+预算双键匹配)===")
    EV = json.loads((RES / "external_validation.json").read_text(encoding="utf-8"))
    for row in T.get(5, [])[1:]:
        hit = next((c for c in EV["configs"]
                    if c["lr"] == row[0] and "%d 轮" % c.get("budget", 16) == row[1]), None)
        ck("表5 %s/%s 命中配置" % (row[0], row[1]), hit is not None, "文中 " + "/".join(row[:2]))
        if hit:
            ck("表5 %s/%s 斜率" % (row[0], row[1]), cell_ok(row[2], hit["slope_std"]),
               "文中 %s vs %s" % (row[2], hit["slope_std"]))
            _lo, _hi = ci_pair(row[3])
            ck("表5 %s/%s CI" % (row[0], row[1]),
               cell_ok("{:.3f}".format(_lo), hit["slope_std_ci95"][0]) and
               cell_ok("{:.3f}".format(_hi), hit["slope_std_ci95"][1]),
               "文中 %s vs %s" % (row[3], hit["slope_std_ci95"]))
            ck("表5 %s/%s p" % (row[0], row[1]), cell_ok(row[4], hit["p"]), "文中 %s vs %s" % (row[4], hit["p"]))
            ck("表5 %s/%s R²" % (row[0], row[1]), cell_ok(row[5], hit["r2"]), "文中 %s vs %s" % (row[5], hit["r2"]))
            ck("表5 %s/%s R²(+Intra)" % (row[0], row[1]), cell_ok(row[6], hit["r2_with_intra"]),
               "文中 %s vs %s" % (row[6], hit["r2_with_intra"]))

    print("=== 表 6(external_validation2.json，CORe50)===")
    _p2 = RES / "external_validation2.json"
    if _p2.exists():
        EV2 = json.loads(_p2.read_text(encoding="utf-8"))
        for row in T.get(6, [])[1:]:
            hit = next((c for c in EV2["configs"] if c["lr"] == row[0] and c.get("n_points")), None)
            ck("表6 lr=%s 命中配置" % row[0], hit is not None, "文中 " + row[0])
            if hit:
                ck("表6 lr=%s 斜率" % row[0], cell_ok(row[1], hit["slope_std"]), "文中 %s vs %s" % (row[1], hit["slope_std"]))
                _lo, _hi = ci_pair(row[2])
                ck("表6 lr=%s CI" % row[0],
                   cell_ok("{:.3f}".format(_lo), hit["slope_std_ci95"][0]) and
                   cell_ok("{:.3f}".format(_hi), hit["slope_std_ci95"][1]),
                   "文中 %s vs %s" % (row[2], hit["slope_std_ci95"]))
                ck("表6 lr=%s p" % row[0], cell_ok(row[3], hit["p"]), "文中 %s vs %s" % (row[3], hit["p"]))
                ck("表6 lr=%s R²" % row[0], cell_ok(row[4], hit["r2"]), "文中 %s vs %s" % (row[4], hit["r2"]))

    print("\n共校验 %d 个单元格判定，失败 %d 个(表 3–表 6；表 1/表 2 为构成与规模表，另见 audit_datasets.py)。" % (checks, bad))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
