# Daily bars 2017-12-01 .. now + split/dividend events for the universe (Yahoo chart API), 4 worker threads (as the original yhd_fetch.py).
# Raw responses saved gzip-compressed and untouched (untrusted; parsed later with json only). Resumable. usage: yhd_fetch2.py UNIVERSE_CSV OUT_DIR
import sys, os, time, gzip, threading, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
UNI, OUT = sys.argv[1], sys.argv[2]; REFRESH = len(sys.argv) > 3 and sys.argv[3] == "refresh"; P1, P2 = 1512086400, int(time.time())
uni = pd.read_csv(UNI, keep_default_na=False)
todo = [(r.symbol, r.yahoo) for r in uni.itertuples() if REFRESH or (not os.path.exists(os.path.join(OUT, r.symbol + ".json.gz")) and not os.path.exists(os.path.join(OUT, r.symbol + ".err")))]
lock = threading.Lock(); stat = {"ok": 0, "fail": 0, "r429": 0}; t0 = time.time()
def work(item):
    sym, y = item; s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0 (rally-research-script)"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y}?period1={P1}&period2={P2}&interval=1d&events=div%2Csplits"
    for attempt in range(6):
        try:
            r = s.get(url, timeout=40)
            if r.status_code == 200:
                tmp = os.path.join(OUT, sym + ".json.gz.part"); open(tmp, "wb").write(gzip.compress(r.content)); os.replace(tmp, os.path.join(OUT, sym + ".json.gz"))
                with lock: stat["ok"] += 1
                break
            if r.status_code == 429:
                with lock: stat["r429"] += 1
                time.sleep(10 * (attempt + 1)); continue
            open(os.path.join(OUT, sym + ".err"), "w").write(str(r.status_code))
            with lock: stat["fail"] += 1
            break
        except Exception:
            time.sleep(4)
    time.sleep(0.25)
    with lock:
        n = stat["ok"] + stat["fail"]
        if n % 500 == 0: print(n, "/", len(todo), stat, round(time.time() - t0), "s", flush=True)
with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(work, todo))
print("done", stat, round(time.time() - t0), "s", flush=True)
