# -*- coding: utf-8 -*-
"""四案例对照：域内基线、各档 Δacc、逐域斜率、合并斜率、ΔFréchet 单项 R²。

用于回答"OfficeHome 与本文三案例的相同点/冲突点"。
输出：results/case_contrast.json + 终端表
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CASES = [("pacs", 128, 40, ["", "_strong", "_lr1e4"]),
         ("organ", 64, 24, ["", "_strong", "_lr1e4"]),
         ("digits", 64, 16, ["", "_strong", "_lr1e4"]),
         ("officehome", 128, 16, ["", "_strong", "_lr1e4"])]
TAGL = {"": "3e-4", "_strong": "1e-3", "_lr1e4": "1e-4"}


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def fit(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum() if len(y) > 2 else float("nan")
    return float(b[1]), float(r2)


out = {}
for case, size, ep, tags in CASES:
    for tag in tags:
        p = RES / ("%s_resnet18_s%d%s.csv" % (case, size, tag))
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        base = {}
        for r in rows:
            if int(r["epoch"]) == 0:
                base[(int(r["seed"]), r["domain"])] = (float(r["acc_ridge"]), float(r["fr_dist"]))
        doms = sorted({r["domain"] for r in rows})
        # 域内基线(epoch0 精度均值)
        inb = float(np.mean([v[0] for k, v in base.items() if "域内" in k[1]]))
        # 末轮 Δacc(相对基线)
        d = {}
        for dm in doms:
            vals = [100 * (float(r["acc_ridge"]) - base[(int(r["seed"]), dm)][0])
                    for r in rows if int(r["epoch"]) == ep and r["domain"] == dm]
            d[dm] = float(np.mean(vals)) if vals else float("nan")
        # 合并斜率(域外点，标准化；按域×轮取种子均值)
        pts = {}
        for r in rows:
            e, sd = int(r["epoch"]), int(r["seed"])
            if e == 0 or "域内" in r["domain"]:
                continue
            a0, f0 = base[(sd, r["domain"])]
            pts.setdefault((r["domain"], e), {})[sd] = (100 * (float(r["acc_ridge"]) - a0),
                                                        float(r["fr_dist"]) - f0)
        xs = [np.mean([v[1] for v in pts[k].values()]) for k in sorted(pts)]
        ys = [np.mean([v[0] for v in pts[k].values()]) for k in sorted(pts)]
        pooled, r2p = fit(z(xs), z(ys))
        # 逐域斜率(域内跨轮，标准化)
        per = {}
        for dm in doms:
            if "域内" in dm:
                continue
            xx = [pts[(dm, e)][s][1] for e in sorted({k[1] for k in pts if k[0] == dm})
                  for s in pts[(dm, e)]]
            yy = [pts[(dm, e)][s][0] for e in sorted({k[1] for k in pts if k[0] == dm})
                  for s in pts[(dm, e)]]
            s_, r2_ = fit(z(xx), z(yy))
            per[dm.split("(")[0]] = round(s_, 2)
        out["%s%s" % (case, tag)] = dict(case=case, tag=TAGL[tag], in_domain_base=round(inb, 3),
                                        delta=dict((k.split("(")[0], round(v, 2)) for k, v in d.items()),
                                        pooled_slope=round(pooled, 3), pooled_r2=round(r2p, 3),
                                        per_domain=per)

print("=" * 100)
print("四案例对照(10 种子；Δacc 为末轮相对未微调基线，单位 pp)")
print("=" * 100)
print("  %-11s %-6s %8s %10s %10s %12s %10s  %s" % (
    "案例", "强度", "域内基线", "域内Δ", "域外Δ(均)", "合并斜率", "单项R²", "逐域斜率"))
for k, v in out.items():
    ood = [x for dm, x in v["delta"].items() if dm != "Real World" and dm != "photo"
           and dm != "axial" and dm != "mnist"]
    inkey = [dm for dm in v["delta"] if dm in ("photo", "axial", "mnist", "Real World")][0]
    print("  %-11s %-6s %8.3f %10.2f %10.2f %12.3f %10.3f  %s" % (
        v["case"], v["tag"], v["in_domain_base"], v["delta"][inkey], float(np.mean(ood)),
        v["pooled_slope"], v["pooled_r2"],
        "  ".join("%s %+.2f" % (a, b) for a, b in v["per_domain"].items())))
(RES / "case_contrast.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n已写出:", RES / "case_contrast.json")
