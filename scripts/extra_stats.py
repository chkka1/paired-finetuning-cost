# -*- coding: utf-8 -*-
"""补充统计(评审报告 C1/C2/C3)：

C1  主效应的标准差、95% 置信区间与显著性检验(种子数 5；配对 t 检验)
C2  “先量距离、再决定是否微调”判据的离线验证(始终微调 / 从不微调 / 按判据)
C3  案例内标准化后的距离-代价相关，以及 MMD 与 Fréchet 两种诊断量的对比

数据来源：results/*.csv(逐轮、逐域、逐种子的原始结果)，不依赖 analysis.json。
输出：results/extra_stats.json(供成稿使用)+ 终端摘要。
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TAG_FROM_NAME = re.compile(r"_s\d+(?P<tag>.*)$")
SKIP = ("_smoke",)
# 每个案例的“源域”与其设定顺序(弱→强)
SOURCE = {"pacs": "photo", "organ": "axial", "digits": "mnist"}
TAG_ORDER = ["_lr1e4", "", "_strong"]


def dom_name(raw_dom):
    """CSV 里的域名形如 'art_painting(域外)' -> 'art_painting'"""
    return raw_dom.split("(")[0]


def is_in(raw_dom):
    """源域(域内)判定：CSV 中域名带 '(域内)' 后缀。"""
    return "域内" in raw_dom


MAIN_SIZE = {"pacs": 128, "organ": 64, "digits": 64}   # 主序列尺寸；对照实验(224/ViT)另行分析


def is_main(row) -> bool:
    """只保留主序列(ResNet-18 + 各案例主尺寸)，避免对照实验混入论文数字。"""
    return row.get("backbone") == "resnet18" and int(row.get("size", -1)) == MAIN_SIZE.get(row["case"], -1)


def load():
    """返回 raw[(case, tag, seed, domain)][epoch] = 该轮记录。"""
    raw = defaultdict(dict)
    for p in sorted(RESULTS.glob("*.csv")):
        if any(k in p.name for k in SKIP):
            continue
        tag = TAG_FROM_NAME.search(p.stem).group("tag")
        case = p.name.split("_")[0]
        with p.open(encoding="utf-8") as f:
            for row in csv_rows(f):
                row["case"] = case
                if not is_main(row):
                    continue
                key = (case, tag, int(row["seed"]), row["domain"])
                raw[key][int(row["epoch"])] = row
    return raw


def csv_rows(f):
    import csv
    return list(csv.DictReader(f))


def deltas(raw):
    """d[(case, tag, seed, domain)] = {epoch: Δacc(pp)}, 以同 seed 同域的 epoch 0 为基线。"""
    d = {}
    for key, by_ep in raw.items():
        base = float(by_ep[0]["acc_ridge"]) if 0 in by_ep else None
        if base is None:
            continue
        d[key] = {e: 100.0 * (float(r["acc_ridge"]) - base) for e, r in by_ep.items()}
    return d


def final_epoch(raw, case, tag):
    for key, by_ep in raw.items():
        if key[0] == case and key[1] == tag:
            return max(by_ep)
    return None


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return None, None
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return float(x.mean() - h), float(x.mean() + h)


def c1_main_effects(raw, d):
    """关键增量的均值±标准差、95% CI 与对 0 的单样本 t 检验；以及强-弱配对检验。"""
    out = {"effects": [], "paired": []}
    for case in ("pacs", "organ", "digits"):
        for tag in TAG_ORDER:
            keys = [(k, v) for k, v in raw.items() if k[0] == case and k[1] == tag]
            if not keys:
                continue
            ep = final_epoch(raw, case, tag)
            seeds = sorted({k[2] for k, _ in keys})
            for dom in sorted({k[3] for k, _ in keys}):
                vals = [d[(case, tag, s, dom)][ep] for s in seeds if (case, tag, s, dom) in d]
                if len(vals) < 2:
                    continue
                lo, hi = ci95(vals)
                t, p = stats.ttest_1samp(vals, 0.0)
                out["effects"].append({
                    "case": case, "tag": tag, "domain": dom, "role": "in" if is_in(dom) else "out",
                    "epoch": ep, "n_seed": len(vals),
                    "mean": round(float(np.mean(vals)), 2), "sd": round(float(np.std(vals, ddof=1)), 2),
                    "ci95": [round(lo, 2), round(hi, 2)], "p_vs0": float(p),
                })
    # 强 vs 弱(同一案例、同一轮数、同一域，按种子配对)
    for case in ("pacs", "organ", "digits"):
        if not any(k[0] == case and k[1] == "_strong" for k in raw):
            continue
        ep = final_epoch(raw, case, "_strong")
        seeds = sorted({k[2] for k in raw if k[0] == case and k[1] == "_strong"})
        for dom in sorted({k[3] for k in raw if k[0] == case}):
            a = [d.get((case, "_strong", s, dom), {}).get(ep) for s in seeds]
            b = [d.get((case, "", s, dom), {}).get(ep) for s in seeds]
            pair = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(pair) < 2:
                continue
            xs, ys = zip(*pair)
            t, p = stats.ttest_rel(xs, ys)
            out["paired"].append({
                "case": case, "domain": dom, "role": "in" if is_in(dom) else "out", "epoch": ep,
                "strong_mean": round(float(np.mean(xs)), 2), "weak_mean": round(float(np.mean(ys)), 2),
                "diff": round(float(np.mean(np.asarray(xs) - np.asarray(ys))), 2),
                "ci95_diff": [round(v, 2) for v in ci95(np.asarray(xs) - np.asarray(ys))],
                "p_paired": float(p), "n_pair": len(pair),
            })
    return out


def c2_decision_rule(raw, d):
    """判据验证：用第 1 轮的 ΔMMD 决定是否继续微调，比较三种策略的域外/域内结果。"""
    recs = []
    for key, by_ep in raw.items():
        case, tag, seed, dom = key
        if is_in(dom):
            continue
        if 1 not in by_ep or 0 not in by_ep:
            continue
        ep = max(by_ep)
        dmmd1 = float(by_ep[1]["mmd"]) - float(by_ep[0]["mmd"])
        dacc = d.get(key, {}).get(ep)
        if dacc is None:
            continue
        recs.append({"case": case, "tag": tag, "domain": dom, "d_mmd1": dmmd1,
                     "d_acc_final": dacc, "seeds": seed})
    always = np.array([r["d_acc_final"] for r in recs], dtype=float)
    never = np.zeros_like(always)
    rule = np.array([r["d_acc_final"] if r["d_mmd1"] <= 0 else 0.0 for r in recs], dtype=float)
    # 域内：按判据不微调时，域内增益记为 0
    ind = []
    for key, by_ep in raw.items():
        case, tag, seed, dom = key
        if not is_in(dom):
            continue
        if 1 not in by_ep:
            continue
        ep = max(by_ep)
        dmmd1_any = None
        for k2, b2 in raw.items():          # 该案例该轮次下任一目标域的 ΔMMD(各域共用一次决策)
            if k2[0] == case and k2[1] == tag and k2[2] == seed and k2[3] != dom and 1 in b2:
                dmmd1_any = float(b2[1]["mmd"]) - float(b2[0]["mmd"])
                break
        g = d.get(key, {}).get(ep)
        if g is None:
            continue
        ind.append((g, 0.0 if (dmmd1_any is not None and dmmd1_any > 0) else g))
    ind_always = np.array([a for a, _ in ind], dtype=float)
    ind_rule = np.array([b for _, b in ind], dtype=float)
    def summarize(a):
        return {"mean": round(float(a.mean()), 2), "sd": round(float(a.std(ddof=1)), 2) if len(a) > 1 else None,
                "n": int(len(a))}
    # 按案例细分
    per_case = {}
    for c in ("pacs", "organ", "digits"):
        sel = [r for r in recs if r["case"] == c]
        if not sel:
            continue
        a = np.array([r["d_acc_final"] for r in sel], dtype=float)
        ru = np.array([r["d_acc_final"] if r["d_mmd1"] <= 0 else 0.0 for r in sel], dtype=float)
        per_case[c] = {"always": round(float(a.mean()), 2), "never": 0.0,
                       "rule": round(float(ru.mean()), 2), "n": len(sel),
                       "rule_kept": int(sum(1 for r in sel if r["d_mmd1"] <= 0))}
    return {"n_obs": len(recs), "out_always": summarize(always), "out_never": summarize(never),
            "out_rule": summarize(rule), "in_always": summarize(ind_always), "in_rule": summarize(ind_rule),
            "per_case": per_case}


def c3_diagnostics(raw, d):
    """案例内标准化后的相关 + MMD/Fréchet 对比。"""
    def collect(metric):
        pts = defaultdict(list)         # case -> [(d_metric, d_acc, role)]
        for key, by_ep in raw.items():
            case, tag, seed, dom = key
            for e, row in by_ep.items():
                if e == 0:
                    continue
                dd = d.get(key, {}).get(e)
                if dd is None:
                    continue
                dm = float(row[metric]) - float(by_ep[0][metric])
                pts[case].append((dm, dd, "in" if is_in(dom) else "out"))
        return pts

    out = {}
    for name, metric in (("mmd", "mmd"), ("frechet", "fr_dist")):
        pts = collect(metric)
        # 案例内标准化后合并(消除案例间尺度差异 → 缓解辛普森效应)
        zx, zy = [], []
        per_case = {}
        for case, arr in pts.items():
            arr = [p for p in arr if p[2] == "out"]
            if len(arr) < 3:
                continue
            x = np.array([p[0] for p in arr]); y = np.array([p[1] for p in arr])
            sx, sy = x.std(ddof=0), y.std(ddof=0)
            if sx > 0 and sy > 0:
                zx += list((x - x.mean()) / sx); zy += list((y - y.mean()) / sy)
            r, p = stats.pearsonr(x, y)
            rs, ps = stats.spearmanr(x, y)
            per_case[case] = {"n": int(len(arr)), "r": round(float(r), 3), "p": float(p),
                              "spearman": round(float(rs), 3)}
        r_all, p_all = stats.pearsonr(zx, zy) if len(zx) > 3 else (float("nan"), float("nan"))
        out[name] = {"n": len(zx), "r_standardized": round(float(r_all), 3), "p_standardized": float(p_all),
                     "per_case": per_case}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    raw = load()
    d = deltas(raw)
    res = {"n_rows": sum(len(v) for v in raw.values()),
           "c1": c1_main_effects(raw, d), "c2": c2_decision_rule(raw, d), "c3": c3_diagnostics(raw, d)}
    (RESULTS / "extra_stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== C1 关键效应(末轮，均值±标准差，95% CI，对 0 检验)===")
    for e in res["c1"]["effects"]:
        if e["role"] == "in" or abs(e["mean"]) >= 3:
            print("  %-6s %-8s %-14s Δ=%+6.2f±%.2f  CI[%+.2f, %+.2f]  p=%.3g"
                  % (e["case"], e["tag"] or "3e-4", e["domain"], e["mean"], e["sd"],
                     e["ci95"][0], e["ci95"][1], e["p_vs0"]))
    print("\n=== C1 强-弱配对检验(同轮数、同域、按种子配对)===")
    for e in res["c1"]["paired"]:
        print("  %-6s %-14s 强%+6.2f 弱%+6.2f 差%+6.2f CI[%+.2f, %+.2f] p=%.3g"
              % (e["case"], e["domain"], e["strong_mean"], e["weak_mean"], e["diff"],
                 e["ci95_diff"][0], e["ci95_diff"][1], e["p_paired"]))
    print("\n=== C2 判据验证(域外，n=%d)===" % res["c2"]["n_obs"])
    for k, lab in (("out_always", "始终微调"), ("out_never", "从不微调"), ("out_rule", "按判据")):
        v = res["c2"][k]; print("  %s: 域外 Δacc 均值 %+.2f pp(n=%d)" % (lab, v["mean"], v["n"]))
    print("  域内：始终微调 %+.2f pp → 按判据 %+.2f pp"
          % (res["c2"]["in_always"]["mean"], res["c2"]["in_rule"]["mean"]))
    for c, v in res["c2"]["per_case"].items():
        print("    %-6s 始终 %+.2f | 从不 0.00 | 按判据 %+.2f(保留微调 %d/%d)"
              % (c, v["always"], v["rule"], v["rule_kept"], v["n"]))
    print("\n=== C3 诊断量对比(案例内标准化后的总体相关)===")
    for name, v in res["c3"].items():
        print("  %-8s n=%d r=%+.3f p=%.2e | 分案例 %s" % (
            name, v["n"], v["r_standardized"], v["p_standardized"],
            {c: d2["r"] for c, d2 in v["per_case"].items()}))
    print("\n已写出:", RESULTS / "extra_stats.json")


if __name__ == "__main__":
    main()
