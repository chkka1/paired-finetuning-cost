# -*- coding: utf-8 -*-
"""复现性审计：
  A. 数据与缓存是否齐备(原始数据源、缓存指纹)
  B. 分析链是否确定性可复现(重跑 4 个分析脚本，比较 JSON 哈希)
  C. 训练是否可复现(重跑 1 个种子的 1 轮，与已存 CSV 的 epoch-0/epoch-1 对照)
  D. reproduce.ps1 是否覆盖全部实际使用的运行
输出：终端报告 + results/reproducibility.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
out: dict = {"data": [], "analysis": [], "training": {}, "reproduce_coverage": {}}


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "(缺失)"


print("=" * 96)
print("A. 原始数据与缓存")
print("=" * 96)
raws = {
    "PACS(HF flwrlabs/pacs)": ROOT / "data" / "pacs.parquet",
    "OfficeHome(HF flwrlabs/office-home)": ROOT / "data" / "officehome" / "data",
    "CORe50(HF adrake17/core50)": ROOT / "data" / "core50" / "data",
    "MedMNIST/数字(torchvision 下载缓存)": ROOT / "data" / "raw",
}
for name, p in raws.items():
    if p.is_dir():
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
        print("  %-38s 目录：%3d 文件 %8.1f MB" % (name, n, mb))
        out["data"].append(dict(name=name, kind="dir", files=n, mb=round(mb, 1)))
    else:
        print("  %-38s %s" % (name, ("%.1f MB" % (p.stat().st_size / 1e6)) if p.exists() else "**缺失**"))
        out["data"].append(dict(name=name, kind="file", exists=p.exists()))
print("\n  缓存(data/cases/*.npz)：")
for f in sorted((ROOT / "data" / "cases").glob("*.npz")):
    print("    %-16s %8.1f MB  sha256=%s" % (f.name, f.stat().st_size / 1e6, h(f)))

print("\n" + "=" * 96)
print("B. 分析链确定性(重跑并比较 JSON 哈希)")
print("=" * 96)
CHAIN = ["analyze.py", "extra_stats.py", "theory_fit.py", "cross_case_model.py",
         "regularities.py", "external_validation.py", "external_validation_core50.py"]
targets = {"analyze.py": "analysis.json", "extra_stats.py": "extra_stats.json",
           "theory_fit.py": "theory_fit.json", "cross_case_model.py": "cross_case.json",
           "regularities.py": "regularities.json", "external_validation.py": "external_validation.json",
           "external_validation_core50.py": "external_validation2.json"}
for s in CHAIN:
    tgt = RES / targets[s]
    before = h(tgt)
    r = subprocess.run([PY, str(ROOT / "scripts" / s)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    after = h(tgt)
    same = before == after
    print("  %-28s exit=%s  %s → %s  %s" % (s, r.returncode, before, after,
                                            "一致 ✓" if same else "**不一致** ✗"))
    out["analysis"].append(dict(script=s, exit=r.returncode, before=before, after=after, identical=same))

print("\n" + "=" * 96)
print("C. 训练可复现性(重跑 officehome / seed 0 / 1 轮，与已存 CSV 对照)")
print("=" * 96)
smoke_tag = "_repro"
smoke_csv = RES / ("officehome_resnet18_s128%s.csv" % smoke_tag)
if smoke_csv.exists():
    smoke_csv.unlink()
r = subprocess.run([PY, str(ROOT / "scripts" / "run_experiment.py"), "--case", "officehome",
                    "--size", "128", "--epochs", "1", "--eval-epochs", "1", "--seeds", "0",
                    "--batch", "64", "--lr", "3e-4", "--tag", smoke_tag],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
main_csv = RES / "officehome_resnet18_s128.csv"
def rows(p):
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
if smoke_csv.exists() and main_csv.exists():
    sm = {(int(x["seed"]), int(x["epoch"]), x["domain"]): x for x in rows(smoke_csv)}
    mn = {(int(x["seed"]), int(x["epoch"]), x["domain"]): x for x in rows(main_csv)}
    print("  域                      epoch0 基线(重跑 vs 已存)        epoch1 精度差")
    for k in sorted(sm):
        if k not in mn:
            continue
        a, b = sm[k], mn[k]
        d0 = abs(float(a["acc_ridge"]) - float(b["acc_ridge"])) if k[1] == 0 else None
        d1 = abs(float(a["acc_ridge"]) - float(b["acc_ridge"])) if k[1] == 1 else None
        print("  %-22s %s   %s" % (k[2], ("%.4f vs %.4f(差 %.4f)" % (
            float(a["acc_ridge"]), float(b["acc_ridge"]), d0)) if d0 is not None else "—",
            ("%.4f vs %.4f(差 %.4f)" % (float(a["acc_ridge"]), float(b["acc_ridge"]), d1)) if d1 is not None else "—"))
        out["training"]["%s|%d" % (k[2], k[1])] = dict(seed0_rerun=float(a["acc_ridge"]),
                                                       stored=float(b["acc_ridge"]))
    print("  (epoch 0 为未微调基线，应完全一致；epoch 1 允许 GPU 非确定性带来的末位差异)")
    smoke_csv.unlink()
else:
    print("  冒烟运行失败：exit=%s %s" % (r.returncode, r.stderr[-300:]))

print("\n" + "=" * 96)
print("D. reproduce.ps1 覆盖度(实际用到的运行 vs 脚本中记录)")
print("=" * 96)
rp = ROOT / "reproduce.ps1"
txt = rp.read_text(encoding="utf-8", errors="replace") if rp.exists() else ""
need = {
    "officehome(16 轮 ×3 档)": "--case officehome",
    "officehome 40 轮(_e40)": "--tag _e40",
    "core50(官方测试会话)": "--case core50",
    "core50all(全 11 会话)": "--case core50all",
    "seeds 5-9(扩种子)": '"5","6","7","8","9"',
    "batch=64 统一": '"--batch","64"',
}
for name, pat in need.items():
    ok = pat.replace('"', "") in txt.replace('"', "")
    print("  %-30s %s" % (name, "已记录 ✓" if ok else "**未记录** ✗"))
    out["reproduce_coverage"][name] = bool(ok)
(RES / "reproducibility.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n已写出:", RES / "reproducibility.json")
