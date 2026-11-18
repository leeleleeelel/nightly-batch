# Daily panel 2017-12-01..latest as float32 arrays [session, symbol] (O, H, L, C, V) from one fresh Yahoo download (split-adjusted quotes,
# as the original panel). Calendar = INTC sessions (as build_panel_x19.py). Also writes splits, a per-symbol report and Yahoo meta fields.
# Raw files are untrusted: parsed with json only. usage: build_panel2.py RAW_DIR UNIVERSE_CSV CORE
import sys, os, gzip, json, numpy as np, pandas as pd
RAW, UNI, CORE = sys.argv[1:4]; PX = os.path.join(CORE, "px"); os.makedirs(PX, exist_ok=True); os.makedirs(os.path.join(CORE, "meta"), exist_ok=True)
uni = pd.read_csv(UNI, keep_default_na=False)
def parse(sym):
    p = os.path.join(RAW, sym + ".json.gz")
    if not os.path.exists(p): return None, [], {}
    try: r = json.loads(gzip.decompress(open(p, "rb").read()))["chart"]["result"][0]
    except Exception: return None, [], {}
    ts = r.get("timestamp") or []; q = (r.get("indicators", {}).get("quote") or [{}])[0]; meta = r.get("meta", {}) or {}
    spl = [dict(symbol=sym, date=pd.to_datetime(int(v["date"]), unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d"),
                numerator=v.get("numerator"), denominator=v.get("denominator"), ratio=v.get("splitRatio")) for v in ((r.get("events") or {}).get("splits") or {}).values()]
    if not ts: return None, spl, meta
    d = pd.to_datetime(np.array(ts, dtype="int64"), unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    df = pd.DataFrame({"date": d, **{k[0]: pd.to_numeric(pd.Series(q.get(k, [None] * len(ts))), errors="coerce").values for k in ["open", "high", "low", "close", "volume"]}})
    df.columns = ["date", "o", "h", "l", "c", "v"]
    return df.dropna(subset=["c"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True), spl, meta
data, spl_rows, metas = {}, [], []
for s in uni.symbol:
    df, sp, mt = parse(s); spl_rows += sp
    if df is not None and len(df): data[s] = df
    metas.append(dict(symbol=s, exchangeName=mt.get("exchangeName"), fullExchangeName=mt.get("fullExchangeName"), instrumentType=mt.get("instrumentType"),
                      firstTradeDate=pd.to_datetime(mt.get("firstTradeDate"), unit="s", errors="coerce") if mt.get("firstTradeDate") else pd.NaT, currency=mt.get("currency")))
cal = pd.DatetimeIndex(sorted(set(data["INTC"].date))); N = len(cal); syms = [s for s in uni.symbol if s in data]; S = len(syms)
A = {c: np.full((N, S), np.nan, dtype=np.float32) for c in "ohlcv"}; pos = pd.Series(np.arange(N), index=cal); rep = []
for j, s in enumerate(syms):
    D = data[s]; D = D[D.date.isin(pos.index)]; k = pos[D.date].to_numpy()
    for c in "ohlcv": A[c][k, j] = D[c].to_numpy(dtype=np.float32)
    rep.append(dict(symbol=s, first=str(D.date.min().date()) if len(D) else "", n_bars=len(D)))
for c in "ohlcv": np.save(os.path.join(PX, f"{c}.npy"), A[c])
np.save(os.path.join(PX, "cal.npy"), cal.values); json.dump(syms, open(os.path.join(PX, "syms.json"), "w"))
pd.DataFrame(spl_rows).drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"]).to_csv(os.path.join(PX, "splits_x19.csv"), index=False)
pd.DataFrame(rep).to_csv(os.path.join(PX, "panel_report.csv"), index=False); pd.DataFrame(metas).to_csv(os.path.join(CORE, "meta", "meta.csv"), index=False)
print("sessions", N, cal[0].date(), cal[-1].date(), "symbols", S, "of", len(uni), "| splits", len(spl_rows))
