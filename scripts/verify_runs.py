# -*- coding: utf-8 -*-
"""运行结果查证(等结果齐备后执行)。

用途：对"扩种子 + 统一 batch + OfficeHome 40 轮后续"后的结果做系统核验，输出：
  A. 完整性：种子 × 评估轮次 × 域 覆盖；重复行；n 是否恒定；主干/尺寸是否符合预期；epoch 0 基线是否齐备
  B. 10 种子 vs 5 种子：主系列三案例 + OfficeHome 的 Δacc(均值±SD)，结论是否随种子数变化
  C. batch 128(归档)vs batch 64：同种子配对比较(Organ/Digits)，量化批大小的实际影响
  D. OfficeHome 16 轮 vs 40 轮：同种子配对比较(同一 epoch 与末轮)，检验"预算不足"这一替代解释
  E. 外部验证主检验在 10 种子与 40 轮下的斜率/CI/判定

用法：python scripts/verify_runs.py            # 打印报告并写出 results/verification.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ARCH = RES / "archive_b128"

# 各案例的预期设置(与正文 2.4 / 表 2 一致)
EXPECT = {
    "pacs":       dict(size=128, epochs=40, evals=[1, 2, 4, 8, 16, 24, 32, 40], src="photo"),
    "organ":      dict(size=64,  epochs=24, evals=[1, 2, 4, 8, 16, 24], src="axial"),
    "digits":     dict(size=64,  epochs=16, evals=[1, 2, 4, 8, 16], src="mnist"),
    "officehome": dict(size=128, epochs=16, evals=[1, 2, 4, 8, 16], src="Real World"),
}
TAGS = {"": "1e-4/3e-4/1e-3 之 3e-4(base)", "_strong": "1e-3", "_lr1e4": "1e-4"}
EXTRA = {"_e40": ("officehome", 40), "_strong_e40": ("officehome", 40), "_lr1e4_e40": ("officehome", 40)}
out: dict = {"A_integrity": [], "B_seeds": {}, "C_batch": {}, "D_budget": {}, "E_primary": {}}


def rows_of(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_tag(p: Path) -> tuple[str, str]:
    """从文件名解析 (case, tag)：<case>_resnet18_s<size><tag>.csv"""
    stem = p.stem
    head = stem.split("_resnet18_s", 1)
    case = head[0]
    rest = head[1]                     # 例如 "128_strong"
    tag = rest.split("_", 1)[1] if "_" in rest else ""
    return case, ("_" + tag if tag else "")


def curve(rows: list[dict], seeds: list[int] | None = None):
    """返回 {(domain, epoch): {seed: (Δacc, ΔFréchet, ΔIntra)}} 与基线字典。"""
    base = {}
    for r in rows:
        if int(r["epoch"]) == 0:
            base[(int(r["seed"]), r["domain"])] = (float(r["acc_ridge"]), float(r["fr_dist"]),
                                                   float(r["intra"]), float(r["n"]))
    pts = defaultdict(dict)
    for r in rows:
        e, sd = int(r["epoch"]), int(r["seed"])
        if e == 0 or (seeds is not None and sd not in seeds):
            continue
        k = (sd, r["domain"])
        if k not in base:
            continue
        a0, f0, i0, _ = base[k]
        pts[(r["domain"], e)][sd] = (100 * (float(r["acc_ridge"]) - a0),
                                     float(r["fr_dist"]) - f0, float(r["intra"]) - i0)
    return pts, base


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def fit(x, y):
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = max(len(y) - 2, 1)
    se = np.sqrt((res ** 2).sum() / dof / ((np.asarray(x) - np.mean(x)) ** 2).sum())
    return float(b[1]), float(se)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 88)
    print("A. 完整性核验(期望 10 种子 0–9)")
    print("=" * 88)
    for p in sorted(RES.glob("*.csv")):
        if any(k in p.name for k in ("s224", "vit", "smoke")):
            continue
        case, tag = split_tag(p)
        rows = rows_of(p)
        if not rows:
            continue
        exp = EXPECT.get(case)
        seeds = sorted({int(r["seed"]) for r in rows})
        eps = sorted({int(r["epoch"]) for r in rows})
        doms = sorted({r["domain"] for r in rows})
        keys = [(int(r["seed"]), int(r["epoch"]), r["domain"]) for r in rows]
        dup = len(keys) - len(set(keys))
        ns = sorted({float(r["n"]) for r in rows})
        ok_seed = seeds == list(range(10))
        exp_ev = exp["evals"] if exp and not tag.startswith("_e40") and not tag.endswith("_e40") else (
            [1, 2, 4, 8, 16, 24, 32, 40] if tag.endswith("e40") else (exp["evals"] if exp else []))
        ok_ep = eps == [0] + list(exp_ev)
        has_base = all((sd, d) in {(int(r["seed"]), r["domain"]) for r in rows if int(r["epoch"]) == 0}
                       for sd in seeds for d in doms)
        print("  %-38s 种子 %s | 轮次 %s | 域 %d | 重复 %d | n %s | 基线齐 %s" % (
            p.name[:38], "10✓" if ok_seed else "仅%d" % len(seeds),
            "✓" if ok_ep else "%d 个" % len(eps), len(doms), dup,
            ("恒定 %g" % ns[0]) if len(ns) == 1 else "变动 %s" % ns[:3], "✓" if has_base else "✗"))
        out["A_integrity"].append(dict(file=p.name, case=case, tag=tag, seeds=len(seeds), epochs=eps,
                                       domains=len(doms), dup=dup, n_constant=len(ns) == 1,
                                       baseline_complete=has_base, backbone=rows[0]["backbone"],
                                       size=int(rows[0]["size"])))
    print("\n" + "=" * 88)
    print("B. 10 种子 vs 5 种子(主系列三案例；Δacc 均值±SD，末轮)")
    print("=" * 88)
    for case in ("pacs", "organ", "digits", "officehome"):
        exp = EXPECT[case]
        for tag in ("", "_strong", "_lr1e4"):
            p = RES / ("%s_resnet18_s%d%s.csv" % (case, exp["size"], tag))
            rows = rows_of(p)
            if not rows:
                continue
            last = max(EXPECT[case]["evals"])
            pts, base = curve(rows)
            line = []
            for seeds in ([0, 1, 2, 3, 4], list(range(10))):
                vals = {}
                for (d, e), dd in pts.items():
                    if e != last:
                        continue
                    v = [dd[s][0] for s in dd if s in seeds]
                    if v:
                        vals[d.split("(")[0]] = (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0)
                line.append(vals)
            if len(line) == 2:
                txt = "；".join("%s %+.2f→%+.2f (SD %.2f→%.2f)" % (
                    k, line[0].get(k, (np.nan,))[0], line[1].get(k, (np.nan,))[0],
                    line[0].get(k, (np.nan, 0))[1], line[1].get(k, (np.nan, 0))[1]) for k in line[1])
                print("  %-11s %-8s %s" % (case, tag or "3e-4", txt))
                out["B_seeds"]["%s%s" % (case, tag or "_base")] = {
                    "epoch": last, "five": {k: list(map(float, v)) for k, v in line[0].items()},
                    "ten": {k: list(map(float, v)) for k, v in line[1].items()}}
    print("\n" + "=" * 88)
    print("C. batch 128(归档，种子 0–4)vs batch 64(新，同种子配对)")
    print("=" * 88)
    for case in ("organ", "digits"):
        exp = EXPECT[case]
        for tag in ("", "_strong", "_lr1e4"):
            old = rows_of(ARCH / ("%s_resnet18_s%d%s.csv" % (case, exp["size"], tag)))
            new = rows_of(RES / ("%s_resnet18_s%d%s.csv" % (case, exp["size"], tag)))
            if not old or not new:
                continue
            po, _ = curve(old, seeds=[0, 1, 2, 3, 4])
            pn, _ = curve(new, seeds=[0, 1, 2, 3, 4])
            last = max(exp["evals"])
            txt, diffs = [], []
            for d in sorted({k[0] for k in pn}):
                vo = [po[(d, last)][s][0] for s in sorted(po[(d, last)]) if s in po.get((d, last), {})]
                vn = [pn[(d, last)][s][0] for s in sorted(pn[(d, last)])]
                m = min(len(vo), len(vn))
                if m >= 2:
                    diffs.append(np.mean(vn[:m]) - np.mean(vo[:m]))
                    tt = stats.ttest_rel(vn[:m], vo[:m])
                    txt.append("%s %+.2f→%+.2f (配对 p=%.3f)" % (d.split("(")[0], np.mean(vo[:m]),
                                                               np.mean(vn[:m]), tt.pvalue))
            if txt:
                print("  %-8s %-8s %s" % (case, tag or "3e-4", "；".join(txt)))
                out["C_batch"]["%s%s" % (case, tag or "_base")] = dict(epoch=last, per_domain=txt)
    print("\n" + "=" * 88)
    print("D. OfficeHome 16 轮 vs 40 轮(同种子配对，末轮)")
    print("=" * 88)
    for tag16, tag40 in (("", "_e40"), ("_strong", "_strong_e40"), ("_lr1e4", "_lr1e4_e40")):
        r16 = rows_of(RES / ("officehome_resnet18_s128%s.csv" % tag16))
        r40 = rows_of(RES / ("officehome_resnet18_s128%s.csv" % tag40))
        if not r16 or not r40:
            print("  %-10s 待结果(40 轮文件未就绪)" % (tag16 or "3e-4"))
            continue
        p16, _ = curve(r16, seeds=list(range(10)))
        p40, _ = curve(r40, seeds=list(range(10)))
        for want16, want40, note in ((16, 16, "同为第 16 轮(仅余弦调度视界不同：T_max=16 vs 40)"),
                                     (16, 40, "各自末轮(16 轮 vs 40 轮，即完整预算对照)")):
            txt = []
            for d in sorted({k[0] for k in p40}):
                a = p16.get((d, want16), {})
                b = p40.get((d, want40), {})
                ss = sorted(set(a) & set(b))
                if len(ss) >= 2:
                    va = [a[s][0] for s in ss]
                    vb = [b[s][0] for s in ss]
                    tt = stats.ttest_rel(vb, va)
                    txt.append("%s %+.2f→%+.2f (配对 p=%.3f)" % (d.split("(")[0], np.mean(va), np.mean(vb), tt.pvalue))
            if txt:
                print("  %-10s %s：%s" % (tag16 or "3e-4", note, "；".join(txt)))
                out["D_budget"]["%s_e%d_vs_e%d" % (tag16 or "base", want16, want40)] = txt
    print("\n" + "=" * 88)
    print("E. 外部验证主检验(OfficeHome，域外点，按域×轮取种子均值)")
    print("=" * 88)
    for label, fn, evals in (("16 轮 / 10 种子(原主检验)", "officehome_resnet18_s128.csv", EXPECT["officehome"]["evals"]),
                             ("40 轮 / 10 种子(附加条件)", "officehome_resnet18_s128_e40.csv", [1, 2, 4, 8, 16, 24, 32, 40])):
        rows = rows_of(RES / fn)
        if not rows:
            print("  %-26s 待结果" % label)
            continue
        pts, _ = curve(rows, seeds=list(range(10)))
        doms = sorted({d for d, _ in pts if "域内" not in d})
        eps = [e for e in evals if e != 0]
        y, x = [], []
        for d in doms:
            for e in eps:
                if (d, e) in pts and len(pts[(d, e)]) >= 2:
                    y.append(np.mean([v[0] for v in pts[(d, e)].values()]))
                    x.append(np.mean([v[1] for v in pts[(d, e)].values()]))
        s, se = fit(z(x), z(y))
        crit = float(stats.t.ppf(0.975, max(len(y) - 2, 1)))
        pv = 2 * float(stats.t.sf(abs(s / se), max(len(y) - 2, 1)))
        print("  %-26s 斜率 %+.3f  95%% CI [%+.3f, %+.3f]  p=%.3f  n=%d" % (
            label, s, s - crit * se, s + crit * se, pv, len(y)))
        out["E_primary"][label] = dict(slope=round(s, 3), ci=[round(s - crit * se, 3), round(s + crit * se, 3)],
                                       p=round(pv, 4), n=len(y))
    (RES / "verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出:", RES / "verification.json")


if __name__ == "__main__":
    main()
