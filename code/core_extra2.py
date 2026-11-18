# Rebuild of build_core_extra.py: svr.npy (FINRA daily short-volume ratio, sessions x symbols), si.pkl (short interest), xbrl_facts.pkl
# (XBRL frames with availability = filing acceptance time, else period end + 60 days). Inputs are read-only. usage: core_extra2.py DATA CORE WORK
import sys, os, glob, gzip, json, numpy as np, pandas as pd
D, CORE, W = sys.argv[1:4]
cal = pd.DatetimeIndex(np.load(CORE + "/px/cal.npy")); syms = json.load(open(CORE + "/px/syms.json")); col = {s: j for j, s in enumerate(syms)}
pos = pd.Series(np.arange(len(cal)), index=cal)
SV = np.full((len(cal), len(syms)), np.nan, dtype=np.float32)
for ch in pd.read_csv(D + "/finra/shvol.csv.gz", dtype={"Symbol": str}, keep_default_na=False, chunksize=2_000_000):
    ch["d"] = pd.to_datetime(ch.Date.astype(str)); x = ch[ch.d.isin(pos.index) & ch.Symbol.isin(col)].copy()
    x["TotalVolume"] = pd.to_numeric(x.TotalVolume, errors="coerce"); x["ShortVolume"] = pd.to_numeric(x.ShortVolume, errors="coerce"); x = x[x.TotalVolume > 0]
    SV[pos[x.d].to_numpy(), x.Symbol.map(col).to_numpy()] = (x.ShortVolume / x.TotalVolume).to_numpy(dtype=np.float32)
np.save(CORE + "/svr.npy", SV); print("svr filled share", round(float(np.isfinite(SV).mean()), 3), flush=True); del SV
si = pd.read_csv(D + "/finra/si.csv.gz", dtype={"symbolCode": str}, keep_default_na=False)
si = si.rename(columns={"symbolCode": "symbol", "currentShortPositionQuantity": "si", "daysToCoverQuantity": "dtc", "changePercent": "si_chg", "averageDailyVolumeQuantity": "adv"})
for c in ("si", "dtc", "si_chg", "adv"): si[c] = pd.to_numeric(si[c], errors="coerce").astype("float32")
si["settle"] = pd.to_datetime(si.settle.astype(str)); si = si[si.symbol.isin(col)].drop_duplicates(["symbol", "settle"]).sort_values(["symbol", "settle"]).reset_index(drop=True)
si.to_pickle(CORE + "/si.pkl"); print("si rows", len(si), si.settle.min().date(), si.settle.max().date(), flush=True)
F = pd.read_pickle(W + "/x19/filings_x19.pkl")[["acc", "accept_et"]]; acc_t = dict(zip(F.acc, F.accept_et)); del F
rows = []
for p in glob.glob(D + "/sec/xbrl_raw/*.json.gz"):
    tag = os.path.basename(p).split("_CY")[0]
    for d in json.loads(gzip.decompress(open(p, "rb").read())).get("data", []):
        rows.append((int(d["cik"]), tag, acc_t.get(d["accn"], pd.Timestamp(d["end"]) + pd.Timedelta(days=60)), pd.Timestamp(d["end"]), float(d["val"])))
X = pd.DataFrame(rows, columns=["cik", "tag", "avail", "end", "val"]); X["tag"] = X.tag.astype("category"); X.to_pickle(CORE + "/xbrl_facts.pkl")
print("xbrl facts", len(X), X.tag.value_counts().to_dict(), flush=True)
