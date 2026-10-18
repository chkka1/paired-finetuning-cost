# -*- coding: utf-8 -*-
"""统计三个案例的构成(域、类别数、train/val/test 样本量)，供正文表 1 使用。

输出：results/dataset_stats.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "cases"
RESULTS = ROOT / "results"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    for name in ("pacs", "organ", "digits", "officehome"):
        p = CASES / (name + ".npz")
        if not p.exists():
            print("missing", p)
            continue
        z = np.load(p, allow_pickle=True)
        labels, domains, splits = z["labels"], z["domains"], z["splits"]
        dnames = [str(x) for x in z["domain_names"]]
        doms = {}
        for d, dn in enumerate(dnames):
            m = domains == d
            doms[dn] = {"n": int(m.sum()),
                        "n_train": int((m & (splits == 0)).sum()),
                        "n_val": int((m & (splits == 1)).sum()),
                        "n_test": int((m & (splits == 2)).sum()),
                        "n_classes": len(set(labels[m].tolist()))}
        out[name] = {"domains": doms, "n_classes": len(set(labels.tolist())),
                     "n_total": int(len(labels)), "n_domains": len(dnames)}
        print("{}: {} 域, {} 类, {} 张".format(name, len(dnames), out[name]["n_classes"], len(labels)))
        for dn, v in doms.items():
            print("   {:<10} n={:<7} train/val/test={}/{}/{}".format(
                dn, v["n"], v["n_train"], v["n_val"], v["n_test"]))
    (RESULTS / "dataset_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print("saved results/dataset_stats.json")


if __name__ == "__main__":
    main()
