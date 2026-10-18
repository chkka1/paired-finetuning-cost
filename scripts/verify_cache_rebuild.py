# -*- coding: utf-8 -*-
"""决定性复现验证：备份现有缓存 → 由原始数据重建 → 比对哈希与内容。
仅重建 pacs / organ / digits(OfficeHome / CORe50 的原始数据一直在库，另行比对)。"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(str(Path(__file__).resolve().parents[1]))
PY = str(R / "venv" / "Scripts" / "python.exe")
CASES = R / "data" / "cases"
BAK = CASES / "_baseline"
BAK.mkdir(exist_ok=True)


def h(p: Path) -> str:
    hh = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            hh.update(b)
    return hh.hexdigest()[:16]


for c in ("pacs", "organ", "digits"):
    src = CASES / ("%s.npz" % c)
    if src.exists() and not (BAK / src.name).exists():
        shutil.copy2(src, BAK / src.name)
        print("  已备份 %s(%.1f MB)" % (src.name, src.stat().st_size / 1e6))

print("\n=== 重建(原始数据来自 data/ 与 data/raw/)===")
res = {}
for c in ("pacs", "organ", "digits"):
    before = h(CASES / ("%s.npz" % c)) if (CASES / ("%s.npz" % c)).exists() else "(无)"
    r = subprocess.run([PY, str(R / "scripts" / "prepare_data.py"), c],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    after = h(CASES / ("%s.npz" % c)) if (CASES / ("%s.npz" % c)).exists() else "(无)"
    out = (r.stdout or "") + (r.stderr or "")
    skipped = ("缺少" in out) or ("!!" in out and "缺少" in out) or ("跳过" in out)
    ok = (before == after) and not skipped
    tail = [l for l in out.split("\n") if l.strip()][-3:]
    res[c] = dict(exit=r.returncode, before=before, after=after, identical=ok,
                  skipped=bool(skipped), tail=tail)
    verdict = ("一致 ✓✓" if (before == after and not skipped)
               else ("**静默跳过(原始数据缺失)** ✗" if skipped else "**不一致** ✗"))
    print("\n  [%s] exit=%s  备份哈希 %s → 重建哈希 %s  %s" % (
        c, r.returncode, before, after, verdict))
    for l in tail:
        print("      %s" % l[:120])

(R / "results" / "reproducibility_rebuild.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n  结果已写出：results/reproducibility_rebuild.json")
print("  若某案例不一致，可从 data/cases/_baseline/ 还原该 npz。")
