import sys, os, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
URL = "https://zenodo.org/api/records/7711810/files/EuroSAT_MS.zip/content"
DST = str(Path(__file__).resolve().parents[1] / "data\eurosat\EuroSAT_MS.zip")
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
UA = {"User-Agent": "Mozilla/5.0 (research-reproducibility)"}
start = time.time()
got = os.path.getsize(DST) if os.path.exists(DST) else 0
req = urllib.request.Request(URL, headers=UA)
if got:
    req.add_header("Range", "bytes=%d-" % got)
try:
    with op.open(req, timeout=120) as r, open(DST, "ab" if got else "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) + got
        n = got
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b); n += len(b)
            if n % (100 << 20) < (1 << 20):
                print("  %.0f / %.0f MB  (%.1f MB/s)" % (n/1e6, total/1e6, n/1e6/(time.time()-start+1e-9)), flush=True)
    print("完成：%.0f MB, 用时 %.0f s" % (os.path.getsize(DST)/1e6, time.time()-start))
except Exception as e:
    print("下载中断：%s（已下载 %.0f MB，可重跑续传）" % (str(e)[:70], (os.path.getsize(DST)/1e6 if os.path.exists(DST) else 0)))
