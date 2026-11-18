# SEC data for the universe (same sources as the original sec_fetch.py / sec_pages_fetch.py / fetch_frames_x19.py), saved raw (gzip) and untouched:
#  1) ticker -> CIK from company_tickers.json (+ EDGAR entity search for the rest, as the original), 2) submissions JSON per CIK + older pages back to
#  2017-06, 3) XBRL frames for 7 concepts, CY2017Q3..CY2026Q3. Generic UA (no personal e-mail), <= 4 requests/s. Resumable.
# usage: sec_fetch2.py UNIVERSE_CSV SEC_DIR
import sys, os, csv, gzip, json, time, requests, pandas as pd
UNI, OUT = sys.argv[1], sys.argv[2]
s = requests.Session(); s.headers.update({"User-Agent": "rally-research-script/1.0 (personal research)", "Accept-Encoding": "gzip, deflate"}); last = [0.0]
def get(u):
    for a in range(4):
        dt = time.time() - last[0]
        if dt < 0.26: time.sleep(0.26 - dt)
        last[0] = time.time()
        try:
            r = s.get(u, timeout=60)
            if r.status_code == 200: return r
            if r.status_code in (429, 503): time.sleep(15 * (a + 1)); continue
            return r
        except Exception: time.sleep(5)
    return None
uni = pd.read_csv(UNI, keep_default_na=False); t0 = time.time()
mp_path = os.path.join(OUT, "cik_map.csv"); COLS = ["symbol", "cik", "entity", "tickers", "status"]
def tickers():
    r = get("https://www.sec.gov/files/company_tickers.json"); tick = {}
    if r is not None and r.status_code == 200:
        for v in json.loads(r.content).values(): tick.setdefault(str(v["ticker"]).upper(), (str(v["cik_str"]), v.get("title", "")))
    return tick
def lookup(pairs, tick):
    rows = []
    for sym, y in pairs:
        hit = tick.get(y.upper()) or tick.get(sym.upper())
        if hit: rows.append([sym, hit[0], hit[1], y, "ok"]); continue
        q = y; rr = get("https://efts.sec.gov/LATEST/search-index?keysTyped=" + requests.utils.quote(q)); cik, ent, tk, st = "", "", "", "nohit"
        if rr is not None and rr.status_code == 200:
            try:
                for h in rr.json().get("hits", {}).get("hits", []):
                    t_ = [x.strip().upper() for x in str(h["_source"].get("tickers", "")).split(",")]
                    if q.upper() in t_ or sym.upper() in t_: cik, ent, tk, st = h["_id"], h["_source"].get("entity", ""), h["_source"].get("tickers", ""), "ok"; break
            except Exception: st = "parse_err"
        rows.append([sym, cik, ent, tk, st])
    return rows
if not os.path.exists(mp_path):
    pd.DataFrame(lookup(zip(uni.symbol, uni.yahoo), tickers()), columns=COLS).to_csv(mp_path, index=False)
else:   # symbols listed after the map was made: company_tickers.json when reachable (not from GitHub runners), else the EDGAR entity search
    old = pd.read_csv(mp_path, dtype=str, keep_default_na=False); have = set(old.symbol)
    todo = [(a, b) for a, b in zip(uni.symbol, uni.yahoo) if a not in have][:400]
    if todo:
        got = [r_ for r_ in lookup(todo, tickers()) if r_[4] == "ok"]   # misses are retried on the next run
        if got: pd.concat([old, pd.DataFrame(got, columns=COLS)], ignore_index=True).to_csv(mp_path + ".part", index=False); os.replace(mp_path + ".part", mp_path)
        print("new symbols", len(todo), "mapped", len(got), [g[0] for g in got][:20], flush=True)
mp = pd.read_csv(mp_path, dtype=str, keep_default_na=False); ciks = sorted(set(int(c) for c in mp.cik if c))
print("mapped", int((mp.status == "ok").sum()), "of", len(mp), "ciks", len(ciks), round(time.time() - t0), "s", flush=True)
n = 0
for c in ciks:
    p = os.path.join(OUT, "subs", f"CIK{c:010d}.json.gz")
    if not os.path.exists(p):
        r = get(f"https://data.sec.gov/submissions/CIK{c:010d}.json")
        if r is None or r.status_code != 200: continue
        open(p, "wb").write(gzip.compress(r.content)); j = json.loads(r.content)
    else:
        j = json.loads(gzip.decompress(open(p, "rb").read()))
    for f in j.get("filings", {}).get("files", []):
        q = os.path.join(OUT, "subs", f["name"] + ".gz")
        if f.get("filingTo", "9999") >= "2017-06-01" and not os.path.exists(q):
            r2 = get("https://data.sec.gov/submissions/" + f["name"])
            if r2 is not None and r2.status_code == 200: open(q, "wb").write(gzip.compress(r2.content))
    n += 1
    if n % 250 == 0: print("subs", n, "/", len(ciks), round(time.time() - t0), "s", flush=True)
Q = [f"CY{y}Q{q}" for y in range(2017, 2027) for q in (1, 2, 3, 4)][2:-1]
C = [("dei", "EntityCommonStockSharesOutstanding", "shares", "I"), ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD", "I"),
     ("us-gaap", "StockholdersEquity", "USD", "I"), ("us-gaap", "Revenues", "USD", ""), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", ""),
     ("us-gaap", "NetIncomeLoss", "USD", ""), ("us-gaap", "Liabilities", "USD", "I")]
for tx, tag, unit, inst in C:
    for q in Q:
        p = os.path.join(OUT, "xbrl_raw", f"{tag}_{q}{inst}.json.gz")
        if os.path.exists(p) or os.path.exists(p + ".404"): continue
        r = get(f"https://data.sec.gov/api/xbrl/frames/{tx}/{tag}/{unit}/{q}{inst}.json")
        if r is not None and r.status_code == 200: open(p, "wb").write(gzip.compress(r.content))
        elif r is not None and r.status_code == 404: open(p + ".404", "w").write("404")
    print("frames", tag, round(time.time() - t0), "s", flush=True)
print("done", round(time.time() - t0), "s", flush=True)
