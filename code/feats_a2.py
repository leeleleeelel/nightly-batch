# REBUILD 2026-11-18: feats_x19_a.py reading the gzip XBRL frames (data/sec/xbrl_raw). W = data/work, S = data.
# Decision-time features (usable at the t0+1 open) for ALL events 2019..2026, computed the same way in every year:
#  SEC filing metadata (as features_sec.py / features_pre60.py / hidden_feats H4 / driver offer_before30), XBRL point-in-time fundamentals
#  (as xbrl_feats.py) + 1-year share growth, raw (unadjusted) price and split history, the 11-commonality profile score at t0 and t0-1,
#  company static flags. History-length-dependent counts are capped so 2019 events and 2026 events see the same look-back.
import sys, os, json, glob, numpy as np, pandas as pd
W, PX = sys.argv[1], sys.argv[2]; S = W[: -len("/work")] if W.endswith("/work") else W
E = pd.read_pickle(W + "/x19/events_x19.pkl")
cal = pd.DatetimeIndex(np.load(PX + "/cal.npy")); N = len(cal); syms = json.load(open(PX + "/syms.json"))
A = {c: np.ascontiguousarray(np.load(PX + f"/{c}.npy").T) for c in "clv"}
mp = pd.read_csv(S + "/sec_raw/cik_map.csv", dtype=str, keep_default_na=False); cik = {s: int(c) for s, c in zip(mp.symbol, mp.cik) if c}
C = pd.read_csv(W + "/sec_company.csv", dtype=str, keep_default_na=False); C["cik"] = C.cik.astype(int); C = C.set_index("cik")
F = pd.read_pickle(W + "/x19/filings_x19.pkl"); F["form"] = F.form.astype(str); F["it"] = F["items"].astype(str)
E["cik"] = E.symbol.map(cik)
# ---------- filing category flags (one boolean column per category) ----------
fo, it = F.form, F.it; is8 = fo.str.startswith("8-K"); is86 = fo.str.match(r"^(8-K|6-K)")
OFF = {"S-1", "S-1/A", "F-1", "F-1/A", "S-3", "S-3/A", "F-3", "F-3/A", "424B1", "424B3", "424B4", "424B5", "424B7"}
FL = {"is86": is86, "earn": is8 & it.str.contains("2.02", regex=False), "i101": it.str.contains("1.01", regex=False), "i701": it.str.contains("7.01", regex=False),
      "i801": it.str.contains("8.01", regex=False), "i502": it.str.contains("5.02", regex=False), "i302": it.str.contains("3.02", regex=False),
      "sixk": fo.str.startswith("6-K"), "f425": fo == "425", "d13": fo.str.contains("13D"), "g13": fo.str.contains("13G"), "off": fo.isin(OFF),
      "tk": fo.isin({"424B1", "424B4", "424B5", "424B7"}), "s8": fo.str.startswith("S-8"), "f4": fo == "4", "f144": fo == "144", "i301": it.str.contains("3.01", regex=False),
      "i503": it.str.contains("5.03", regex=False), "i303": is8 & it.str.contains("3.03", regex=False), "def14": fo.isin(["DEF 14A", "PRE 14A"]), "nt": fo.str.startswith("NT "),
      "i201": it.str.contains("2.01", regex=False), "s1": fo.isin(["S-1", "S-1/A", "F-1", "F-1/A"]), "pre14": fo.isin(["PRE 14A", "PRE 14C"]),
      "b4": fo.isin(["424B4", "424B1"]), "b5": fo.isin(["424B5", "424B3", "424B7"]), "effect": fo == "EFFECT", "formD": fo.isin(["D", "D/A"]),
      "k10": fo.isin(["10-K", "20-F"]), "amend86": is86 & fo.str.endswith("/A"),
      "drv_off": fo.isin(["424B1", "424B3", "424B4", "424B5", "424B7", "S-1", "S-1/A", "F-1", "S-3", "F-3"]) | (is8 & it.str.contains("3.02", regex=False))}
FB = pd.DataFrame({k: v.to_numpy() for k, v in FL.items()})
F = F[["cik", "accept_et", "react_pos", "open_pos"]].join(FB)
tsec = F.accept_et.values.astype("datetime64[s]").astype(np.int64)
G = {}
for c, idx in F.groupby("cik").indices.items():
    o = np.argsort(tsec[idx], kind="stable"); ii = idx[o]
    G[c] = dict(t=tsec[ii], rp=F.react_pos.to_numpy()[ii], op=F.open_pos.to_numpy()[ii], **{k: FB[k].to_numpy()[ii] for k in FL})
fpi = set(F.cik[FB.sixk.to_numpy()]) | set()
DAY = 86400
rows = []
for idx, e in E.iterrows():
    f = {"idx": idx}; c = e.cik
    if c == c and int(c) in C.index:
        r = C.loc[int(c)]; cat = r.category
        f.update(egc=float("Emerging" in cat), large_acc=float("Large accelerated" in cat), nonacc=float("Non-accelerated" in cat), src_small=float("Smaller reporting" in cat),
                 sic=r.sic, sicd=r.sicDescription, state=r.biz_state, country=r.biz_country, inc_state=r.stateOfIncorporation)
    g = G.get(int(c)) if c == c else None
    if g is None: rows.append(f); continue
    i = int(e.i); t0s = int(np.datetime64(e.t0, "s").astype(np.int64)); open_s = t0s + 9 * 3600 + 1800
    av = g["op"] <= i + 1; days = np.floor((t0s - g["t"]) / DAY)
    f["fpi"] = float(int(c) in fpi)
    t0n = av & (g["rp"] == i); post = av & (g["rp"] == i + 1); pre5 = av & (g["rp"] >= i - 5) & (g["rp"] < i)
    def cnt(m, k=None): return int((m & g[k]).sum()) if k else int(m.sum())
    f.update(n_t0=cnt(t0n), n8_t0=cnt(t0n, "is86"), earn_t0=float(cnt(t0n, "earn") > 0), i101_t0=float(cnt(t0n, "i101") > 0), i701_t0=float(cnt(t0n, "i701") > 0),
             i801_t0=float(cnt(t0n, "i801") > 0), i502_t0=float(cnt(t0n, "i502") > 0), i302_t0=float(cnt(t0n, "i302") > 0), sixk_t0=float(cnt(t0n, "sixk") > 0),
             f425_t0=float(cnt(t0n, "f425") > 0), d13_t0=float(cnt(t0n, "d13") > 0), n_post=cnt(post), off_post=float(cnt(post, "off") + cnt(post, "i302") > 0),
             n8_post=cnt(post, "is86"), n8_pre5=cnt(pre5, "is86"), earn_pre5=float(cnt(pre5, "earn") > 0))
    W90, W365, W180 = av & (days <= 90), av & (days <= 365), av & (days <= 180)
    f.update(off90=cnt(W90, "off"), off365=cnt(W365, "off"), tk180=cnt(W180, "tk"), s8_365=cnt(W365, "s8"), f4_90=cnt(W90, "f4"), f144_90=cnt(W90, "f144"),
             d13_90=cnt(W90, "d13"), g13_90=cnt(W90, "g13"), n8_90=cnt(W90, "is86"), i301_365=cnt(W365, "i301"), i503_365=cnt(W365, "i503"), rs_def14=cnt(W365, "def14"),
             nt_365=cnt(W365, "nt"), i101_90=cnt(W90, "i101"), i201_365=cnt(W365, "i201"), f425_90=cnt(W90, "f425"))
    ea = av & g["earn"]
    f["days_since_earn"] = float(days[ea].min()) if ea.any() else np.nan
    f["first_filing_days"] = float(min(days[av].max(), 540)) if av.any() else np.nan
    w60 = (g["t"] < open_s) & (g["t"] >= t0s - 60 * DAY)
    for k in ("s1", "i301", "pre14", "b4", "b5", "nt", "effect", "formD", "i302", "i503", "i303", "f144", "f4", "d13", "k10"):
        f[f"{k}_60"] = float((w60 & g[k]).any() if k not in ("i301", "i302", "i503") else (w60 & g[k] & g["is86"]).any())
    w86 = (g["t"] < t0s) & (g["t"] >= t0s - 60 * DAY) & g["is86"]
    if w86.any():
        tt = pd.to_datetime(g["t"][w86], unit="s"); hm = tt.hour * 60 + tt.minute
        f.update(fil_n60=int(w86.sum()), fil_afterhours=float((hm >= 960).mean()), fil_premarket=float((hm < 570).mean()),
                 fil_friday_eve=float(((tt.weekday == 4) & (hm >= 960)).mean()), fil_amend=float(g["amend86"][w86].mean()))
    else: f["fil_n60"] = 0
    rp = g["rp"]; offm = g["drv_off"]
    f["offer_before30"] = float((offm & (rp <= i) & (rp > i - 21)).any())
    f["offer_after5"] = float((offm & (rp > i) & (rp <= i + 5)).any())           # outcome-side, never a feature
    rows.append(f)
X = pd.DataFrame(rows).set_index("idx")
X["distress60"] = ((X[["s1_60", "i301_60", "pre14_60", "b4_60", "nt_60", "i303_60"]].fillna(0).sum(axis=1)) > 0).astype(float)
# ---------- splits, raw price, XBRL ----------
spl = pd.read_csv(PX + "/splits_x19.csv"); spl["date"] = pd.to_datetime(spl.date); spl["f"] = spl.numerator / spl.denominator
SPL = {s: (g.date.to_numpy(), g.f.to_numpy()) for s, g in spl.groupby("symbol")}
Fa = pd.read_pickle(W + "/x19/filings_x19.pkl")[["acc", "accept_et"]]; acc_t = dict(zip(Fa.acc, Fa.accept_et)); del Fa
facts = {}
for p in glob.glob(S + "/sec/xbrl_raw/*.json.gz"):
    tag = os.path.basename(p).split("_CY")[0]; j = json.loads(__import__("gzip").decompress(open(p, "rb").read()))
    for d in j.get("data", []):
        av_ = acc_t.get(d["accn"], pd.Timestamp(d["end"]) + pd.Timedelta(days=60))
        facts.setdefault((tag, int(d["cik"])), []).append((av_, pd.Timestamp(d["end"]), float(d["val"])))
for k in facts: facts[k].sort()
def latest(tag, c, t, before_end=None):
    best = None
    for av_, end, v in facts.get((tag, c), []):
        if av_ >= t: break
        if before_end is not None and end > before_end: continue
        if best is None or end >= best[0]: best = (end, v)
    return best
rows = []
for idx, e in E.iterrows():
    f = {"idx": idx}; d0 = np.datetime64(e.t0)
    dts, fs = SPL.get(e.symbol, (np.array([], dtype="datetime64[ns]"), np.array([])))
    after = dts > d0; f["adj_to_raw"] = float(np.prod(fs[after])) if after.any() else 1.0
    f["raw_price_pre"] = e.price_pre * f["adj_to_raw"]
    back = (dts <= d0) & (dts > d0 - np.timedelta64(365, "D")); back2 = (dts <= d0) & (dts > d0 - np.timedelta64(730, "D"))
    f["rsplit365"] = float((back & (fs < 1)).any()); f["rsplit730"] = float((back2 & (fs < 1)).any()); f["fsplit365"] = float((back & (fs > 1)).any())
    j = int(e.j); i = int(e.i); Cc = A["c"][j][max(0, i - 30):i].astype(float)
    fut = np.ones(len(Cc)); dd = cal[max(0, i - 30):i]
    for sd, sf in zip(dts, fs):
        fut[dd.values < sd] *= sf
    raw30 = Cc * fut; f["days_below1_30"] = int(np.nansum(raw30 < 1.0)); f["days_below5_30"] = int(np.nansum(raw30 < 5.0))
    c = e.cik
    if c == c:
        c = int(c); t = e.t0 + pd.Timedelta(hours=9, minutes=30)
        sh = latest("EntityCommonStockSharesOutstanding", c, t)
        if sh:
            f.update(shares=sh[1], shares_age_d=(e.t0 - sh[0]).days)
            sh1 = latest("EntityCommonStockSharesOutstanding", c, t, before_end=sh[0] - pd.Timedelta(days=300))
            if sh1 and sh1[1] > 0: f["shares_g1y"] = sh[1] / sh1[1]
        for tag, nm in [("CashAndCashEquivalentsAtCarryingValue", "cash"), ("StockholdersEquity", "equity"), ("Liabilities", "liab"),
                        ("Revenues", "rev_q"), ("RevenueFromContractWithCustomerExcludingAssessedTax", "rev_q2"), ("NetIncomeLoss", "ni_q")]:
            x = latest(tag, c, t)
            if x: f[nm] = x[1]
    rows.append(f)
Y = pd.DataFrame(rows).set_index("idx")
Y["rev_q"] = Y.rev_q.fillna(Y.rev_q2) if "rev_q2" in Y else Y.rev_q
Y["mcap"] = Y.raw_price_pre * Y.shares
for a in ["cash", "equity", "rev_q", "ni_q"]: Y[a + "_mcap"] = Y[a] / Y.mcap
Y["runway_q"] = np.where(Y.ni_q < 0, Y.cash / (-Y.ni_q), np.nan)
X = X.join(Y.drop(columns=["rev_q2"], errors="ignore"))
# ---------- profile score (11 commonalities, thresholds fixed in the earlier train mining) at t0 and t0-1 ----------
shares_ser = {}
for (tag, c), L in facts.items():
    if tag == "EntityCommonStockSharesOutstanding": shares_ser[c] = [(av_, v) for av_, end, v in L]
def pscore(j, k, sym, c, cat):
    Cc, Vv = A["c"][j].astype(float), A["v"][j].astype(float)
    if k < 61 or not np.isfinite(Cc[k]): return np.nan
    with np.errstate(divide="ignore", invalid="ignore"): lr = np.log(Cc[1:k + 1] / Cc[:k])
    lr = np.r_[np.nan, lr]                                       # lr[t] for t = 0..k
    dts, fs = SPL.get(sym, (np.array([], dtype="datetime64[ns]"), np.array([])))
    raw = Cc[k] * (float(np.prod(fs[dts > np.datetime64(cal[k])])) if len(fs) else 1.0)
    sh = np.nan
    if c == c:
        for av_, v in sorted(shares_ser.get(int(c), [])):
            if cal.searchsorted(av_) <= k: sh = v
    spk = (np.abs(lr) >= np.log(1.15)).astype(float)
    def sd(w): x = lr[k - w + 1:k + 1]; return np.nanstd(x, ddof=1) if np.isfinite(x).sum() >= w - 1 else np.nan
    mx250 = np.nanmax(Cc[max(0, k - 249):k + 1]); mx60 = np.nanmax(Cc[k - 59:k + 1])
    v20 = np.nanmean(Vv[k - 19:k + 1]); v60 = np.nanmedian(Vv[k - 79:k - 19]) if k >= 80 else np.nan
    feats = [raw <= 2.73, sd(60) >= 0.0644, np.nansum(spk[k - 59:k + 1]) >= 2, Cc[k] / mx250 <= 0.452, (raw * sh <= 1.29e8) if sh == sh else False,
             np.nansum(spk[k - 19:k + 1]) >= 1, Cc[k] / mx60 <= 0.643, "Non-accelerated" in cat, (v20 / v60 >= 2.30) if (v60 == v60 and v60 > 0) else False,
             "Emerging" in cat, sd(20) >= 0.063]
    return float(np.sum([bool(x) for x in feats]))
cats = {int(c): r.category for c, r in C.iterrows()}
X["pscore"] = [pscore(int(e.j), int(e.i), e.symbol, e.cik, cats.get(int(e.cik), "") if e.cik == e.cik else "") for idx, e in E.iterrows()]
X["pscore_pre"] = [pscore(int(e.j), int(e.i) - 1, e.symbol, e.cik, cats.get(int(e.cik), "") if e.cik == e.cik else "") for idx, e in E.iterrows()]
X.to_pickle(W + "/x19/feat_a.pkl")
print(X.shape); print(X.notna().mean().round(2).sort_values().head(12).to_dict())
print(pd.concat([E.year, X[["earn_t0", "large_acc", "off90", "shares", "mcap", "pscore", "pscore_pre", "rsplit365", "days_below1_30"]]], axis=1).groupby("year").mean().round(3).to_string())
