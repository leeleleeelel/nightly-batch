# Rebuild of fetch_ftd.py + fetch_ftd_missing.py: SEC fails-to-deliver half-month zips 2018-06..latest, saved as-is (untrusted; parsed later as text).
# Exact links are read from the SEC data page first (non-standard paths exist). Logs Last-Modified for the availability rule. Generic UA, ~1 req/s.
# usage: fetch_ftd2.py OUT_DIR LOG_CSV
import sys, os, re, csv, time, requests, pandas as pd
OUT, LOG = sys.argv[1], sys.argv[2]
s = requests.Session(); s.headers.update({"User-Agent": "rally-research-script/1.0 (personal research)", "Accept-Encoding": "gzip, deflate"})
page = s.get("https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data", timeout=60).text
links = {}
for href, key in re.findall(r'href="([^"]*cnsfails(\d{6}[ab])(?:_\d)?\.zip)"', page): links.setdefault(key, []).append(href if href.startswith("http") else "https://www.sec.gov" + href)
keys = [p.strftime("%Y%m") + h for p in pd.period_range("2018-06", pd.Timestamp.today().strftime("%Y-%m"), freq="M") for h in "ab"]
done = {r["key"] for r in csv.DictReader(open(LOG, encoding="utf-8")) if r["status"] == "200"} if os.path.exists(LOG) else set()
new = not os.path.exists(LOG); fh = open(LOG, "a", newline="", encoding="utf-8"); w = csv.writer(fh)
if new: w.writerow(["key", "status", "bytes", "last_modified"])
for k in keys:
    if k in done: continue
    urls = links.get(k, []) + [f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{k}.zip"]; got = False
    for url in urls:
        for t in range(3):
            try:
                r = s.get(url, timeout=120)
                if r.status_code == 200 and r.content[:2] == b"PK":
                    open(f"{OUT}/cnsfails{k}.zip", "wb").write(r.content); w.writerow([k, 200, len(r.content), r.headers.get("Last-Modified", "")]); got = True; break
                if r.status_code in (403, 404): break
            except Exception: pass
            time.sleep(3 * (t + 1))
        time.sleep(0.8)
        if got: break
    if not got: w.writerow([k, 404, 0, ""])
    fh.flush()
print("done", len(keys), "page links", len(links))
