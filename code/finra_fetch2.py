# Same sources and reduction as the original finra_fetch.py, extended to the present: FINRA daily short-sale volume (CNMSshvol, 2019-01-02..now)
# and twice-monthly short interest (shrt, 2018-12..now) from cdn.finra.org, reduced on arrival to universe symbols, appended to gzip CSVs.
# Raw text parsed with pandas only (untrusted). Resumable. usage: finra_fetch2.py UNIVERSE_CSV OUT_DIR
import sys, os, io, time, requests, pandas as pd
UNI, OUT = sys.argv[1], sys.argv[2]
syms = set(pd.read_csv(UNI, keep_default_na=False).symbol)
s = requests.Session(); s.headers.update({"User-Agent": "rally-research-script/1.0 (personal research)"})
def get(u):
    for a in range(4):
        try:
            r = s.get(u, timeout=60)
            if r.status_code == 200: return r.text
            if r.status_code in (403, 404): return None
            time.sleep(10 * (a + 1))
        except Exception: time.sleep(5)
    return None
SV = os.path.join(OUT, "shvol.csv.gz"); done_p = os.path.join(OUT, "shvol_done.txt"); done = set(open(done_p).read().split()) if os.path.exists(done_p) else set()
part = []; t0 = time.time(); n = 0
def flush():
    global part
    if part: pd.concat(part).to_csv(SV, mode="a", index=False, header=not os.path.exists(SV), compression="gzip"); part = []
    open(done_p, "w").write("\n".join(sorted(done)))
for d in pd.bdate_range("2019-01-02", pd.Timestamp.today().normalize() - pd.Timedelta(days=1)):
    k = f"{d:%Y%m%d}"
    if k in done: continue
    t = get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{k}.txt")
    if t:
        x = pd.read_csv(io.StringIO(t), sep="|", dtype={"Symbol": str}, usecols=["Date", "Symbol", "ShortVolume", "TotalVolume"], keep_default_na=False)
        part.append(x[x.Symbol.isin(syms)])
    done.add(k); n += 1; time.sleep(0.15)
    if n % 60 == 0: flush(); print("shvol", k, n, round(time.time() - t0), "s", flush=True)
flush()
SI = os.path.join(OUT, "si.csv.gz"); done_p2 = os.path.join(OUT, "si_done.txt"); done2 = set(open(done_p2).read().split()) if os.path.exists(done_p2) else set()
cols = ["symbolCode", "currentShortPositionQuantity", "daysToCoverQuantity", "changePercent", "averageDailyVolumeQuantity"]
for d in pd.date_range("2018-12-01", pd.Timestamp.today().normalize()):
    if d.weekday() >= 5 or not (10 <= d.day <= 17 or d.day >= 25): continue
    k = f"{d:%Y%m%d}"
    if k in done2: continue
    t = get(f"https://cdn.finra.org/equity/otcmarket/biweekly/shrt{k}.csv")
    if t and len(t) > 100000:
        x = pd.read_csv(io.StringIO(t), sep="|", dtype={"symbolCode": str}, usecols=cols, keep_default_na=False); x = x[x.symbolCode.isin(syms)]; x["settle"] = k
        x.to_csv(SI, mode="a", index=False, header=not os.path.exists(SI), compression="gzip"); print("si", k, len(x), flush=True)
    done2.add(k); open(done_p2, "w").write("\n".join(sorted(done2))); time.sleep(0.15)
print("done", round(time.time() - t0), "s", flush=True)
