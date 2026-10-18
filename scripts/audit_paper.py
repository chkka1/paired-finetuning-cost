# -*- coding: utf-8 -*-
"""论文核查(针对新加入的 ViT-B/16@224 实验)：

A. 独立重算表 7(不复用 robustness_check.py 的代码路径)，核对正文/表格数字；
B. 给出三个配置距离-代价斜率的 95% 置信区间，判断“符号不稳健”这一表述的强度；
C. 把正文(不含参考文献)里所有带小数的数字与底层 JSON 的“数字宇宙”对照，列出对不上的；
D. 列出所有含“拉开/拉远/拉近/同向/负相关/普适”等关键词的句子，供人工检查是否过度表述。
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPOCH = 16
SEEDS = (0, 1, 2)
OUT_DOMAINS = ("art_painting", "cartoon", "sketch")
CONFIGS = [("ResNet-18@128", "pacs_resnet18_s128.csv"),
           ("ResNet-18@224", "pacs_resnet18_s224_r224.csv"),
           ("ResNet-18@224b16", "pacs_resnet18_s224_r224b16.csv"),
           ("ResNet-18@224lr4", "pacs_resnet18_s224_r224lr4.csv"),
           ("ViT-B/16@224b16", "pacs_vit_b_16_s224_v224.csv"),
           ("ViT-B/16@224lr4", "pacs_vit_b_16_s224_v224lr4.csv")]


def zk(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def ols(y, xs):
    X = np.column_stack([np.ones(len(y))] + xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = len(y) - X.shape[1]
    cov = (res ** 2).sum() / dof * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov)), 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ---------------- A. 独立重算表 7 ----------------
print("=" * 78)
print("A. 独立重算表 7(先按 (种子,域,轮) 建基线，再算 Δacc 与 ΔFréchet)")
print("=" * 78)
tbl = {}
for label, fname in CONFIGS:
    rows = list(csv.DictReader((RES / fname).open(encoding="utf-8")))
    base = {(int(r["seed"]), r["domain"]): (float(r["acc_ridge"]), float(r["fr_dist"]))
            for r in rows if int(r["epoch"]) == 0}
    pts = defaultdict(list)
    for r in rows:
        ep, sd = int(r["epoch"]), int(r["seed"])
        if ep == 0 or ep > EPOCH or sd not in SEEDS:
            continue
        a0, f0 = base[(sd, r["domain"])]
        pts[(sd, r["domain"])].append((ep, 100 * (float(r["acc_ridge"]) - a0), float(r["fr_dist"]) - f0))
    # 末轮均值
    def endmean(dom):
        vals = [v[-1][1] for k, v in pts.items() if dom.split("(")[0] in k[1]]
        return float(np.mean(vals)) if vals else None
    ind = endmean("photo(域内)")
    outs = {d: endmean(d + "(域外)") for d in OUT_DOMAINS}
    # 斜率：全部 ≤16 轮域外点
    X, Y = [], []
    for (sd, dom), seq in pts.items():
        if "域外" not in dom:
            continue
        X += [s[2] for s in seq]
        Y += [s[1] for s in seq]
    b, se, r2 = ols(zk(Y), [zk(X)])
    tbl[label] = dict(ind=ind, outs=outs, ood=float(np.mean(list(outs.values()))),
                      slope=float(b[1]), se=float(se[1]), r2=float(r2), n=len(X))
    print("  %-14s 域内 %+6.2f | Art %+7.2f Cartoon %+7.2f Sketch %+7.2f | 域外均值 %+7.2f | 斜率 %+.2f (se %.2f, 95%%CI [%+.2f, %+.2f]) R²=%.2f n=%d" % (
        label, ind, outs["art_painting"], outs["cartoon"], outs["sketch"],
        tbl[label]["ood"], tbl[label]["slope"], se[1] and tbl[label]["se"],
        tbl[label]["slope"] - 1.96 * tbl[label]["se"], tbl[label]["slope"] + 1.96 * tbl[label]["se"],
        r2, len(X)))

print("\n  与 robustness.json 比对：")
rb = json.loads((RES / "robustness.json").read_text(encoding="utf-8"))
ok = True
for label, v in tbl.items():
    r = rb["configs"][label]
    checks = [("域内", v["ind"], r["in_domain"]),
              ("Art", v["outs"]["art_painting"], r["out_domains"]["art_painting"]),
              ("Cartoon", v["outs"]["cartoon"], r["out_domains"]["cartoon"]),
              ("Sketch", v["outs"]["sketch"], r["out_domains"]["sketch"]),
              ("域外均值", v["ood"], r["out_mean"]), ("斜率", v["slope"], r["slope"])]
    for name, a, b2 in checks:
        if abs(a - b2) > 0.006:
            print("    ✗ %-14s %-8s 独立重算 %.3f ≠ 文件 %.3f" % (label, name, a, b2))
            ok = False
print("  结论：%s" % ("全部一致 ✓" if ok else "存在不一致 ✗"))

# ---------------- B. 斜率的置信区间(判“符号不稳健”是否站得住) ----------------
print("\n" + "=" * 78)
print("B. 斜率 95% CI(判断“符号不稳健”的强度；注意 45 个点非独立，CI 偏窄)")
print("=" * 78)
for label, v in tbl.items():
    lo, hi = v["slope"] - 1.96 * v["se"], v["slope"] + 1.96 * v["se"]
    sig = "不含 0" if lo * hi > 0 else "含 0(不显著)"
    print("  %-14s 斜率 %+.2f  CI [%+.2f, %+.2f]  %s  R²=%.2f" % (label, v["slope"], lo, hi, sig, v["r2"]))
s128, s224 = tbl["ResNet-18@128"]["slope"], tbl["ResNet-18@224"]["slope"]
d = s224 - s128
sed = np.hypot(tbl["ResNet-18@128"]["se"], tbl["ResNet-18@224"]["se"])
print("  ResNet@128 与 @224 斜率之差 %+.2f，CI [%+.2f, %+.2f] → %s" % (
    d, d - 1.96 * sed, d + 1.96 * sed, "差异显著" if (d - 1.96 * sed) * (d + 1.96 * sed) > 0 else "差异不显著"))

# ---------------- C. 正文数字 vs 数字宇宙 ----------------
print("\n" + "=" * 78)
print("C. 正文带小数的数字是否都能在底层结果中找到")
print("=" * 78)
uni = set()


def addnum(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return
    for nd in (1, 2, 3):
        uni.add(("%+.*f" % (nd, f)).replace("+", ""))
        uni.add("%.*f" % (nd, f))


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        addnum(o)


for f in ("analysis.json", "extra_stats.json", "theory_fit.json", "cross_case.json", "robustness.json"):
    walk(json.loads((RES / f).read_text(encoding="utf-8")))
# 结果 CSV 里的精度/距离值
for p in RES.glob("*.csv"):
    for r in csv.DictReader(p.open(encoding="utf-8")):
        for k in ("acc_ridge", "acc_nc", "mmd", "fr_dist", "intra"):
            addnum(r[k])
print("  数字宇宙规模: %d" % len(uni))

md = (ROOT / "manuscript_final.md").read_text(encoding="utf-8")
body = md.split("## 参考文献")[0]
nums = re.findall(r"[-−]?\d+\.\d+", body)
uni2 = set(uni) | {x.replace("−", "-").lstrip("-") for x in uni} | {x.lstrip("0").lstrip(".") for x in uni}
missing = []
for raw in nums:
    v = raw.replace("−", "-")
    if v.lstrip("-").lstrip("+") in uni2 or v in uni2:
        continue
    missing.append(raw)
print("  正文带小数数字 %d 个，其中 %d 个未在结果中找到：" % (len(nums), len(set(missing))))
for m in sorted(set(missing)):
    ctx = [s for s in re.split(r"[。；\n]", body) if m in s]
    print("    %-8s ← %s" % (m, (ctx[0].strip()[:86] if ctx else "")))

# ---------------- D. 关键词句 ----------------
print("\n" + "=" * 78)
print("D. 含“过度表述”风险关键词的句子")
print("=" * 78)
keys = ("拉开", "拉远", "拉近", "同向", "负相关", "普适", "必然", "总是", "证明")
seen = set()
for s in re.split(r"[。\n]", body):
    s = s.strip()
    if any(k in s for k in keys) and len(s) > 12 and s not in seen:
        seen.add(s)
        print("  - " + s[:150])
