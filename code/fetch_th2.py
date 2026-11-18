# REBUILD 2026-11-18: fetch_th.py over business days 2018-01-02..yesterday (missing days return 404 and are logged).
# Reg SHO threshold lists, one file per session; parsed in memory as pipe-delimited text (untrusted), only symbol/name kept.
# usage: fetch_th.py <nasdaq|nyse> <cal.npy> <out_rows.csv> <out_status.csv>
import sys, os, time, csv, requests, numpy as np, pandas as pd
SRC, CAL, ROWS, STAT = sys.argv[1:5]
days = [d for d in pd.bdate_range("2018-01-02", pd.Timestamp.today().normalize() - pd.Timedelta(days=1))]
s = requests.Session(); s.headers.update({"User-Agent": "rally-research-script/1.0 (personal research)", "Accept-Encoding": "gzip, deflate"})
markets = ["NASDAQ"] if SRC == "nasdaq" else ["NYSE", "NYSE American"]
done = set()
if os.path.exists(STAT):
    done = {(r["date"], r["market"]) for r in csv.DictReader(open(STAT, encoding="utf-8")) if r["status"] == "200"}
newr, news = not os.path.exists(ROWS), not os.path.exists(STAT)
fr = open(ROWS, "a", newline="", encoding="utf-8"); wr = csv.writer(fr); fs = open(STAT, "a", newline="", encoding="utf-8"); ws = csv.writer(fs)
if newr: wr.writerow(["date", "market", "symbol", "name"])
if news: ws.writerow(["date", "market", "status", "n_rows", "trailer"])
def fetch(d, mk):
    if mk == "NASDAQ":
        return s.get(f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{d:%Y%m%d}.txt", timeout=60)
    return s.get("https://www.nyse.com/api/regulatory/threshold-securities/download", params={"selectedDate": d.strftime("%d-%b-%Y"), "market": mk}, timeout=60)
for d in days:
    for mk in markets:
        key = (d.strftime("%Y-%m-%d"), mk)
        if key in done: continue
        st, n, trailer = "ERR", 0, ""
        for t in range(4):
            try:
                r = fetch(d, mk); st = r.status_code
                if r.status_code == 200:
                    lines = [x.strip() for x in r.text.splitlines() if x.strip()]
                    if not lines or not lines[0].startswith("Symbol|"):
                        st = "BADFMT"; break
                    for x in lines[1:]:
                        p = x.split("|")
                        if len(p) >= 4 and p[0]:
                            wr.writerow([key[0], mk, p[0][:12], p[1][:60]]); n += 1
                        elif len(p) == 1 and p[0].isdigit():
                            trailer = p[0]
                    break
                if r.status_code == 404: break
            except Exception as e:
                st = "ERR"
            time.sleep(3 * (t + 1))
        ws.writerow([key[0], mk, st, n, trailer]); fr.flush(); fs.flush(); time.sleep(0.3)
print("done", SRC, len(days))
