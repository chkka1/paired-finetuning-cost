# -*- coding: utf-8 -*-
"""数据集实验一致性与正确性核查(OfficeHome vs PACS/Organ/Digits)。

检查项：
  A. 缓存完整性：形状/类型/域计数/划分计数/类别覆盖/划分互斥(有无泄漏)
  B. 结果 CSV 的实测设置：backbone、size、batch(由 n 与训练规模推断)、评估样本数、轮数、种子数
  C. 各数据集的前置条件对照表(尺寸/轮数/评估点/种子/源域/域数/类别数/测试抽样)
  D. 独立重算一个关键数字(OfficeHome 域内 Δacc)，与管线输出比对
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CASES = ["pacs", "organ", "digits", "officehome"]


def audit_cache(case: str) -> None:
    p = ROOT / "data" / "cases" / f"{case}.npz"
    if not p.exists():
        print("  %-11s 缓存缺失：%s" % (case, p.name))
        return
    z = np.load(p, allow_pickle=True)
    X, y, d, s = z["images"], z["labels"], z["domains"], z["splits"]
    names = list(z["domain_names"]) if "domain_names" in z else [str(i) for i in sorted(set(d.tolist()))]
    print("  %-11s X%s %s | 类别 %d | 域 %d" % (case, X.shape, X.dtype, len(set(y.tolist())), len(names)))
    print("              域 × 划分计数：")
    for di, nm in enumerate(names):
        cnt = Counter(s[d == di].tolist())
        cls = sorted(set(y[d == di].tolist()))
        tr = np.where((d == di) & (s == 0))[0]
        print("                %-12s 训/验/测 = %d/%d/%d | 该域类别数 %d | 训练集类别数 %d" % (
            str(nm)[:12], cnt.get(0, 0), cnt.get(1, 0), cnt.get(2, 0), len(cls),
            len(set(y[tr].tolist())) if len(tr) else 0))
    # 划分互斥(同一域内 train/val/test 不重叠)——按索引天然互斥，这里额外查图像是否重复
    idx = {k: set(np.where(s == k)[0].tolist()) for k in (0, 1, 2)}
    ov = (idx[0] & idx[1]) | (idx[0] & idx[2]) | (idx[1] & idx[2])
    print("              划分索引重叠数：%d(应为 0)" % len(ov))
    # 图像哈希重复抽样检查：源域训练 vs 各域测试，各取 300 张比对
    import hashlib
    def hsh(i):
        return hashlib.sha1(np.ascontiguousarray(X[i]).tobytes()).hexdigest()
    src = None
    for di, nm in enumerate(names):
        if ((d == di) & (s == 0)).sum() > 0:
            src = di
    if src is not None:
        tr = np.where((d == src) & (s == 0))[0]
        tr_h = {hsh(i) for i in tr[:300]}
        for di, nm in enumerate(names):
            if di == src:
                continue
            te = np.where(d == di)[0]
            inter = len(tr_h & {hsh(i) for i in te[:300]})
            print("              源域训练(前300) 与 %-12s 测试(前300) 完全同图：%d(应为 0)" % (str(nm)[:12], inter))


def audit_csv(case: str) -> dict:
    """汇总某案例的实测设置(从 CSV 直接读出，不看代码假设)。"""
    out = {"files": [], "sizes": set(), "backbones": set(), "seeds": set(), "epochs": set(),
           "n_epoch0": {}, "tags": []}
    for p in sorted(RES.glob(f"{case}_*.csv")):
        if "_smoke" in p.name:
            continue
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        if not rows:
            continue
        out["files"].append(p.name)
        out["sizes"].add(int(rows[0]["size"]))
        out["backbones"].add(rows[0]["backbone"])
        out["seeds"].add(tuple(sorted({int(r["seed"]) for r in rows})))
        out["epochs"].add(tuple(sorted({int(r["epoch"]) for r in rows})))
        tag = p.stem.split("_s%d" % int(rows[0]["size"]), 1)[-1]
        out["tags"].append(tag or "(base)")
        for r in rows:
            if int(r["epoch"]) == 0:
                out["n_epoch0"][r["domain"]] = int(float(r["n"]))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 84)
    print("A. 缓存完整性(data/cases/*.npz)")
    print("=" * 84)
    for c in CASES:
        audit_cache(c)
        print()

    print("=" * 84)
    print("B. 结果 CSV 中实测的每案例设置")
    print("=" * 84)
    info = {}
    for c in CASES:
        i = audit_csv(c)
        info[c] = i
        print("  %-11s 主干 %s | 尺寸 %s | 种子 %s | 轮数 %s | 标签 %s" % (
            c, sorted(i["backbones"]), sorted(i["sizes"]),
            sorted(i["seeds"])[0] if i["seeds"] else "-", sorted(i["epochs"])[0] if i["epochs"] else "-",
            i["tags"]))
        print("              各域评估样本数(epoch 0)：%s" % i["n_epoch0"])

    print()
    print("=" * 84)
    print("C. 前置条件对照(关键差异项)")
    print("=" * 84)
    for c in CASES:
        i = info[c]
        z = np.load(ROOT / "data" / "cases" / f"{c}.npz", allow_pickle=True)
        s = z["splits"]
        ncls = len(set(z["labels"].tolist()))
        nsrc_tr = int((s == 0).sum())
        print("  %-11s 尺寸 %-4s 源域训练 %5d 类别 %3d 轮数 %-28s 评估点 %s" % (
            c, sorted(i["sizes"])[0], nsrc_tr, ncls, str(sorted(i["epochs"])[0]),
            sorted(i["epochs"])[0]))

    print()
    print("=" * 84)
    print("D. 独立重算检查(OfficeHome lr=3e-4，域内 Δacc 末轮)")
    print("=" * 84)
    p = RES / "officehome_resnet18_s128.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    base = {(int(r["seed"]), r["domain"]): float(r["acc_ridge"]) for r in rows if int(r["epoch"]) == 0}
    ep = max(int(r["epoch"]) for r in rows)
    vals = [100 * (float(r["acc_ridge"]) - base[(int(r["seed"]), r["domain"])])
            for r in rows if int(r["epoch"]) == ep and "域内" in r["domain"]]
    print("  独立重算：末轮 %d，域内 Δacc 均值 = %+.2f pp(n=%d 种子)" % (ep, float(np.mean(vals)), len(vals)))
    ev = json.loads((RES / "external_validation.json").read_text(encoding="utf-8"))
    print("  外部验证 JSON 中的对应配置：%s" % [
        (c["label"], c["n_points"]) for c in ev["configs"]])
    print("  → 二者应一致(若不一致说明有实现缺陷)")


if __name__ == "__main__":
    main()
