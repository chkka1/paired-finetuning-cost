# -*- coding: utf-8 -*-
"""从 HuggingFace 镜像(hf-mirror.com)下载数据集文件，支持断点续传与多文件并行。

用法：
    python scripts/download_hf_dataset.py <dataset_id> <local_dir> [file1 file2 ...]

不带文件名时下载该数据集全部文件(跳过 .gitattributes)。每文件一个 .part，续传按已写字节。
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 (research-reproducibility)"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api_info(ds: str) -> list[dict]:
    u = "%s/api/datasets/%s?blobs=true" % (MIRROR, urllib.parse.quote(ds))
    with OPENER.open(urllib.request.Request(u, headers=UA), timeout=60) as r:
        j = json.loads(r.read().decode("utf-8"))
    return [s for s in (j.get("siblings") or []) if s["rfilename"] != ".gitattributes"]


def download(ds: str, rfile: str, dst: Path, size: int | None) -> tuple[str, int]:
    """带断点续传与重试的单文件下载。"""
    url = "%s/datasets/%s/resolve/main/%s" % (MIRROR, ds, urllib.parse.quote(rfile))
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    last_err = ""
    for attempt in range(12):
        got = part.stat().st_size if part.exists() else 0
        if size and got >= size:
            part.replace(dst)
            return rfile, got
        if dst.exists() and size and dst.stat().st_size >= size:
            return rfile, dst.stat().st_size
        try:
            req = urllib.request.Request(url, headers=UA)
            if got:
                req.add_header("Range", "bytes=%d-" % got)
            with OPENER.open(req, timeout=90) as r, open(part, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
            if size and got < size:
                raise IOError("提前结束(%d/%d)" % (got, size))
            part.replace(dst)
            return rfile, dst.stat().st_size
        except Exception as e:                      # 超时/中断 → 续传重试
            last_err = str(e)[:60]
            time.sleep(3)
    raise IOError("重试 %d 次仍失败：%s" % (12, last_err))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = sys.argv[1]
    local = ROOT / sys.argv[2]
    wanted = sys.argv[3:]
    files = api_info(ds)
    if wanted:
        files = [f for f in files if f["rfilename"] in wanted]
    total = sum((f.get("size") or 0) for f in files)
    print("数据集 %s：%d 个文件，合计 %.0f MB" % (ds, len(files), total / 1e6))
    results: list[tuple[str, int]] = []
    lock = threading.Lock()

    def run(f: dict) -> None:
        rf = f["rfilename"]
        try:
            res = download(ds, rf, local / rf, f.get("size"))
        except Exception as e:
            res = (rf + " FAILED: " + str(e)[:60], -1)
        with lock:
            results.append(res)
            print("  ✓ %-52s %8.1f MB" % (res[0][:52], res[1] / 1e6), flush=True)

    ths = [threading.Thread(target=run, args=(f,), daemon=True) for f in files]
    t0 = time.time()
    [t.start() for t in ths]
    while any(t.is_alive() for t in ths):
        time.sleep(20)
        got = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
        print("  进度 %.0f / %.0f MB | %.2f MB/s | %.1f 分钟"
              % (got / 1e6, total / 1e6, got / 1e6 / max(time.time() - t0, 1), (time.time() - t0) / 60),
              flush=True)
    for t in ths:
        t.join()
    failed = [r for r in results if r[1] < 0]
    print("完成：%d 个文件成功，%d 个失败%s | 用时 %.1f 分钟"
          % (len(results) - len(failed), len(failed),
             ("(" + "; ".join(r[0] for r in failed) + ")") if failed else "", (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
