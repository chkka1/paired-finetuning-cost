# -*- coding: utf-8 -*-
"""EuroSAT 多光谱版(13 波段)分段并行、可续传下载器。

背景：Zenodo 官方记录(DOI 10.5281/zenodo.7711810)的 EuroSAT_MS.zip 为 2.07 GB，
单线程实测约 7 KB/s、8 路分段约 0.11 MB/s，故采用「分块 + 多线程 + 断点续传」：

- 把文件切成 N 个连续块，每块一个 .part 文件；
- 每个线程只负责自己的块，按已写字节数续传(Range: bytes=start+done-end)；
- 全部块完成后按顺序拼接到最终 zip，并校验总字节数。

用法：python scripts/download_eurosat_parallel.py [线程数]
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/api/records/7711810/files/EuroSAT_MS.zip/content"
TOTAL = 2_065_402_329                      # 官方 Content-Length(HEAD 实测)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eurosat"
DST = OUT / "EuroSAT_MS.zip"
UA = {"User-Agent": "Mozilla/5.0 (research-reproducibility)"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_range(start: int, end: int, part: Path, label: str) -> None:
    """下载 [start, end](含端点)，支持续传。"""
    done = part.stat().st_size if part.exists() else 0
    if done >= (end - start + 1):
        return
    req = urllib.request.Request(URL, headers=dict(UA, Range="bytes=%d-%d" % (start + done, end)))
    with OPENER.open(req, timeout=120) as r, open(part, "ab") as f:
        n = done
        while n < (end - start + 1):
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            n += len(b)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    OUT.mkdir(parents=True, exist_ok=True)
    if DST.exists() and DST.stat().st_size == TOTAL:
        print("已完成，跳过：%s (%.0f MB)" % (DST.name, DST.stat().st_size / 1e6))
        return
    chunk = TOTAL // nthreads
    parts = [OUT / ("EuroSAT_MS.zip.part%d" % i) for i in range(nthreads)]
    bounds = []
    for i in range(nthreads):
        s = i * chunk
        e = TOTAL - 1 if i == nthreads - 1 else (i + 1) * chunk - 1
        bounds.append((s, e))

    errors: list[str] = []
    stop = threading.Event()

    def run(i: int) -> None:
        s, e = bounds[i]
        while not stop.is_set():
            try:
                fetch_range(s, e, parts[i], "part%d" % i)
                if parts[i].stat().st_size >= (e - s + 1):
                    print("  part%d 完成(%.0f MB)" % (i, parts[i].stat().st_size / 1e6), flush=True)
                    return
            except Exception as exc:                      # 网络抖动 → 重试
                errors.append("part%d: %s" % (i, str(exc)[:60]))
                time.sleep(5)

    ths = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(nthreads)]
    t0 = time.time()
    [t.start() for t in ths]
    while any(t.is_alive() for t in ths):
        time.sleep(30)
        got = sum(p.stat().st_size for p in parts if p.exists())
        print("  进度 %.0f / %.0f MB(%.1f%%)| %.2f MB/s | %.0f 分钟"
              % (got / 1e6, TOTAL / 1e6, 100 * got / TOTAL, got / 1e6 / max(time.time() - t0, 1),
                 (time.time() - t0) / 60), flush=True)
    for t in ths:
        t.join()

    if sum(p.stat().st_size for p in parts if p.exists()) != TOTAL:
        print("未完成：块大小合计 %d ≠ %d；可重跑本脚本续传。错误样本：%s"
              % (sum(p.stat().st_size for p in parts if p.exists()), TOTAL, errors[:2]))
        return
    print("拼接中…", flush=True)
    with open(DST, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(8 << 20)
                    if not b:
                        break
                    out.write(b)
    ok = DST.stat().st_size == TOTAL
    print("完成：%s | %.0f MB | 字节数校验 %s" % (DST, DST.stat().st_size / 1e6, "通过" if ok else "失败"))
    if ok:
        for p in parts:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
