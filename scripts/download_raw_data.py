# -*- coding: utf-8 -*-
"""下载 PACS/MedMNIST/数字的原始数据(供第三方从零复现)。

  · PACS   → 由 download_hf_dataset.py 下载(data/pacs.parquet，约 191 MB)
  · 数字   → torchvision MNIST/EMNIST/SVHN + USPS 手工 fallback(prepare_data 的同名函数)
  · MedMNIST(organ) → medmnist 包(Zenodo 源)，若包不可用则给出替代说明
"""
import io
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(str(Path(__file__).resolve().parents[1]))
PY = str(R / "venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(R / "scripts"))

print("=== 1) 数字数据的原始下载(torchvision)===")
try:
    import prepare_data as pd
    root = pd.RAW / "digits"
    root.mkdir(parents=True, exist_ok=True)
    print("  下载目标：", root)
    objs = pd._torchvision_downloads(root)
    for k, v in objs.items():
        try:
            print("   %-12s %d 条" % (k, len(v)))
        except Exception:
            print("   %-12s (已获取)" % k)
    # USPS(证书过期，走手工 fallback)
    try:
        tr, te = pd._usps_manual(root)
        print("   usps         %d + %d 条" % (len(tr), len(te)))
    except Exception as e:
        print("   usps 失败：", str(e)[:80])
except Exception as e:
    print("  数字下载失败：", str(e)[:200])

print("\n=== 2) MedMNIST organ 原始数据 ===")
r = subprocess.run([PY, "-c", "import medmnist;print(medmnist.__version__)"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print("  medmnist 未安装 → 尝试安装(PyPI)")
    r2 = subprocess.run([PY, "-m", "pip", "install", "-q", "medmnist"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("  pip exit=%s" % r2.returncode)
    if r2.returncode != 0:
        print("  安装失败：", (r2.stdout + r2.stderr)[-300:])
r = subprocess.run([PY, "-c", "import medmnist;print('ok', medmnist.__version__)"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("  ", (r.stdout or r.stderr).strip()[:100])
if r.returncode == 0:
    code = ("import medmnist, os; from medmnist import INFO; "
            "d='" + str(R / 'data' / 'raw' / 'medmnist').replace('\\', '\\\\') + "'; os.makedirs(d, exist_ok=True); "
            "from medmnist import OrganAMNIST, OrganCMNIST, OrganSMNIST; "
            "[cls(root=d, download=True, size=28) for cls in (OrganAMNIST, OrganCMNIST, OrganSMNIST)]; "
            "print('organ 三切面已下载到', d)")
    r3 = subprocess.run([PY, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("  ", (r3.stdout or r3.stderr).strip()[-400:])
