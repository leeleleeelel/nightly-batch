# REBUILD 2026-11-18: ftd_feats.py with the threshold-list session range up to the latest session.
# prereg_ftd.json: fails-to-deliver (lagged) and Reg SHO threshold-list (timely) features at t0 for every surge event.
# usage: python -I -X utf8 ftd_feats.py CORE DL_FTD DL_TH   (downloads are untrusted: zip members are read into memory as text only)
import sys, os, re, io, json, zipfile, datetime, numpy as np, pandas as pd
CORE, DLF, DLT = sys.argv[1:4]
cal = pd.to_datetime(np.load(CORE + "/px/cal.npy")); calv = cal.values.astype("datetime64[D]")
syms = json.load(open(CORE + "/px/syms.json")); J = {s: k for k, s in enumerate(syms)}; NS, N = len(syms), len(cal)
C = np.load(CORE + "/px/c.npy"); V = np.load(CORE + "/px/v.npy")
E = pd.read_pickle(CORE + "/events_x19.pkl")[["symbol", "j", "i", "t0"]]
diag = {}

# ---- split factors: raw = adjusted x prod(num/den of splits with ex-date > d); raw shares x factor = adjusted shares
SP = pd.read_csv(CORE + "/px/splits_x19.csv", parse_dates=["date"]); SP = SP[SP.symbol.isin(J)]
SPL = {}
for s, g in SP.sort_values("date").groupby("symbol"):
    r = (g.numerator / g.denominator).to_numpy(float); SPL[J[s]] = (g.date.values.astype("datetime64[D]"), np.cumprod(r[::-1])[::-1])
def factor(j, d):
    """vectorised over d (datetime64[D] array) for one symbol j"""
    d = np.asarray(d, dtype="datetime64[D]")
    if j not in SPL: return np.ones(d.shape)
    ds, cum = SPL[j]; k = np.searchsorted(ds, d, side="right"); out = np.ones(d.shape); m = k < len(ds); out[m] = cum[k[m]]
    return out

# ---- FTD files
log = pd.read_csv(CORE + "/ftd_fetch_log.csv"); log = log[log.status.astype(str) == "200"].drop_duplicates("key", keep="last")
files = sorted(f for f in os.listdir(DLF) if re.fullmatch(r"cnsfails\d{6}[ab]\.zip", f))
keys = [f[8:15] for f in files]
def period_end(k):
    y, m = int(k[:4]), int(k[4:6]); return pd.Timestamp(y, m, 15) if k[6] == "a" else pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
pend = [period_end(k) for k in keys]
lm = dict(zip(log.key.astype(str), pd.to_datetime(log.last_modified, errors="coerce", utc=True)))
lags = []
for k, pe in zip(keys, pend):
    if k >= "202501a" and k in lm and pd.notna(lm[k]):
        lag = (lm[k].tz_convert(None).normalize() - pe).days
        if lag <= 60: lags.append((k, lag))
L = max([21] + [g for _, g in lags]); diag["uniform_L_days_as_first_registered"] = L; diag["lastmod_lags_2025_26"] = lags
lmd = {k: v.tz_convert(None).normalize() for k, v in lm.items() if pd.notna(v)}
parts, created, ikparts = [], [], []
for fi, fn in enumerate(files):
    z = zipfile.ZipFile(os.path.join(DLF, fn)); inf = z.infolist()
    assert len(inf) == 1 and inf[0].file_size < 300e6, fn
    created.append(pd.Timestamp(datetime.datetime(*inf[0].date_time)))
    df = pd.read_csv(io.BytesIO(z.read(inf[0])), sep="|", dtype=str, encoding="latin-1", on_bad_lines="skip", quoting=3)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df[df["SETTLEMENT DATE"].astype(str).str.fullmatch(r"\d{8}", na=False)]
    sym = df["SYMBOL"].astype(str).str.strip().str.upper()
    keep = sym.isin(J)
    d = pd.to_datetime(df.loc[keep, "SETTLEMENT DATE"], format="%Y%m%d")
    q = pd.to_numeric(df.loc[keep, "QUANTITY (FAILS)"], errors="coerce"); p = pd.to_numeric(df.loc[keep, "PRICE"], errors="coerce")
    parts.append(pd.DataFrame(dict(fi=fi, j=sym[keep].map(J).astype(np.int32).to_numpy(), d=d.values.astype("datetime64[D]"), q=q.to_numpy(float), p=p.to_numpy(float))))
    sub = pd.DataFrame(dict(s=sym[keep], iss=df.loc[keep, "CUSIP"].astype(str).str.strip().str[:6],
                            nm=df.loc[keep, "DESCRIPTION"].astype(str).str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True).str.split().str[:2].str.join(" "))).drop_duplicates("s")
    ikparts.append(pd.DataFrame(dict(fi=fi, j=sub.s.map(J).to_numpy(), iss=sub.iss.to_numpy(), nm=sub.nm.to_numpy())))
    if fi % 40 == 0: print("parsed", fn, len(df), int(keep.sum()), flush=True)
R = pd.concat(parts, ignore_index=True); del parts
diag["ftd_rows_ours"] = int(len(R))
crl = [(c.normalize() - pe).days for c, pe in zip(created, pend)]
diag["created_lag_days"] = dict(min=int(min(crl)), median=float(np.median(crl)), max=int(max(crl)))
# availability (amendment_1): never before period end + 21 days, the file's creation date + 1 day, or its posting date (Last-Modified within 120 days)
avail = []
for k, pe, c in zip(keys, pend, created):
    a = max(pe + pd.Timedelta(days=21), c.normalize() + pd.Timedelta(days=1))
    if k in lmd and (lmd[k] - pe).days <= 120: a = max(a, lmd[k])
    avail.append(a)
diag["avail_lag_days"] = pd.Series([(a - pe).days for a, pe in zip(avail, pend)]).describe().round(1).to_dict()
diag["avail_lag_gt30"] = {k: (a - pe).days for k, a, pe in zip(keys, avail, pend) if (a - pe).days > 30}

# identity check per (file, symbol)
prev = np.searchsorted(calv, R.d.values, side="left") - 1
cprev = np.where(prev >= 0, C[np.clip(prev, 0, N - 1), R.j.values], np.nan).astype(float)
fac_prev = np.ones(len(R)); fac_d = np.ones(len(R))
for j, idx in R.groupby("j").indices.items():
    if j in SPL:
        fac_prev[idx] = factor(j, calv[np.clip(prev[idx], 0, N - 1)]); fac_d[idx] = factor(j, R.d.values[idx])
R["ratio"] = R.p.values / (cprev * fac_prev)
R.loc[~np.isfinite(R.ratio) | (R.ratio <= 0), "ratio"] = np.nan
med = R.groupby(["fi", "j"]).ratio.median()
NF = len(files); ST = np.zeros((NF, NS), np.int8)          # 0 absent, 1 match, 2 mismatch, 3 no price to check
fi_, j_ = med.index.get_level_values(0).to_numpy(), med.index.get_level_values(1).to_numpy(); mv = med.to_numpy()
ST[fi_, j_] = np.where(np.isnan(mv), 3, np.where((mv >= 0.85) & (mv <= 1.15), 1, 2))
diag["file_symbol_status_first_pass"] = {k: int((ST == v).sum()) for k, v in (("match", 1), ("mismatch", 2), ("noprice", 3))}
# amendment_3: a price mismatch is often a split missing from our split table, not another company. Walking out from every price-matched
# file of a symbol, an adjacent file of the same symbol is kept while it shares the CUSIP issuer code or the first two name words with the
# file before it; kept mismatch files get the empirical factor (median ratio) as a correction (status 4), kept unverifiable files the
# nearest kept file's correction (status 5). Everything not reached stays rejected (2, 3).
IK = pd.concat(ikparts, ignore_index=True); ISS = np.full((NF, NS), "", object); NM = np.full((NF, NS), "", object)
ISS[IK.fi.values, IK.j.values] = IK.iss.values; NM[IK.fi.values, IK.j.values] = IK.nm.values
MR = np.full((NF, NS), np.nan); MR[fi_, j_] = mv
for j in np.where((ST == 1).any(0) & ((ST == 2) | (ST == 3)).any(0))[0]:
    pres = np.where(ST[:, j] != 0)[0]; acc = ST[pres, j] == 1
    for a in np.where(acc)[0]:
        for step in (-1, 1):
            pi, pn = ISS[pres[a], j], NM[pres[a], j]; k = a + step
            while 0 <= k < len(pres):
                f_ = pres[k]
                if acc[k] or ISS[f_, j] == pi or NM[f_, j] == pn:
                    acc[k] = True; pi, pn = ISS[f_, j], NM[f_, j]; k += step
                else:
                    break
    kept = pres[acc]
    ST[kept[ST[kept, j] == 2], j] = 4; ST[kept[ST[kept, j] == 3], j] = 5
CORR = np.ones((NF, NS)); m4 = ST == 4; CORR[m4] = MR[m4]
for j in np.where((ST == 5).any(0))[0]:
    good = np.where(np.isin(ST[:, j], (1, 4)))[0]
    for f_ in np.where(ST[:, j] == 5)[0]:
        CORR[f_, j] = CORR[good[np.argmin(np.abs(good - f_))], j] if len(good) else 1.0
diag["file_symbol_status"] = {k: int((ST == v).sum()) for k, v in (("match", 1), ("rejected_mismatch", 2), ("rejected_noprice", 3), ("kept_split_corrected", 4), ("kept_unverified", 5))}

# dense settlement-date matrices (only matched rows; mismatched / unverifiable file blocks = NaN)
SD = np.unique(R.d.values); sdi = np.searchsorted(SD, R.d.values)
lo = R.groupby("fi").d.min().reindex(range(NF)); hi = R.groupby("fi").d.max().reindex(range(NF))
FLO = np.searchsorted(SD, lo.values.astype("datetime64[D]")); FHI = np.searchsorted(SD, hi.values.astype("datetime64[D]"))
FA = np.zeros((len(SD), NS), np.float32); FR = np.zeros((len(SD), NS), np.float32); PX = np.zeros((len(SD), NS), np.float32)
ok = np.isin(ST[R.fi.values, R.j.values], (1, 4, 5)); Rm = R[ok]
np.add.at(FA, (sdi[ok], Rm.j.values), (Rm.q.values * fac_d[ok] * CORR[Rm.fi.values, Rm.j.values]).astype(np.float32))
np.add.at(FR, (sdi[ok], Rm.j.values), Rm.q.values.astype(np.float32))
np.maximum.at(PX, (sdi[ok], Rm.j.values), np.nan_to_num(Rm.p.values).astype(np.float32))
for fi in range(NF):
    bad = np.where((ST[fi] == 2) | (ST[fi] == 3))[0]
    if len(bad): FA[FLO[fi]:FHI[fi] + 1, bad] = np.nan; FR[FLO[fi]:FHI[fi] + 1, bad] = np.nan
np.savez_compressed(CORE + "/ftd_dense.npz", FA=FA, FR=FR, PX=PX, SD=SD.astype("datetime64[D]").astype(str), ST=ST, keys=np.array(keys),
                    avail=np.array([a.strftime("%Y-%m-%d") for a in avail]), FLO=FLO, FHI=FHI)
del R, Rm

# shares outstanding (latest XBRL cover-page value filed on or before t0), split-consistent
X = pd.read_pickle(CORE + "/xbrl_facts.pkl"); X = X[X.tag == "EntityCommonStockSharesOutstanding"][["cik", "avail", "end", "val"]].dropna()
X = X[X.val > 0].sort_values(["avail", "end"]).drop_duplicates(["cik", "avail"], keep="last")
cm = pd.read_csv(CORE + "/meta/cik_map.csv"); cm = cm[cm.status == "ok"].drop_duplicates("symbol")
ev = E.reset_index().merge(cm[["symbol", "cik"]], on="symbol", how="left")
ev_ok = ev[ev.cik.notna()].copy(); ev_ok["cik"] = ev_ok.cik.astype(np.int64)
ev_ok = pd.merge_asof(ev_ok.sort_values("t0"), X.rename(columns={"avail": "t0x"}).sort_values("t0x"), left_on="t0", right_on="t0x", by="cik", direction="backward")
ev = ev.merge(ev_ok[["index", "end", "val"]], on="index", how="left").set_index("index").loc[E.index]
tso = np.full(len(E), np.nan)
for k, (j, en, v) in enumerate(zip(ev.j.values, ev.end.values, ev.val.values)):
    if np.isfinite(v): tso[k] = v * factor(int(j), np.array([en], dtype="datetime64[D]"))[0]

# ---- threshold lists (amendment_2: Nasdaq lists only, defined for Nasdaq-listed stocks); a day that failed to download carries the previous day forward
lists = []
for src in ("nasdaq",):
    rows = pd.read_csv(f"{DLT}/{src}_rows.csv", dtype=str, keep_default_na=False); st = pd.read_csv(f"{DLT}/{src}_status.csv", dtype=str)
    st = st.drop_duplicates(["date", "market"], keep="last"); rows = rows.drop_duplicates()
    lists.append((rows, st))
rows = pd.concat([r for r, _ in lists]); stat = pd.concat([s for _, s in lists])
okday = stat[stat.status == "200"]; diag["tl_market_days_ok"] = okday.groupby("market").size().to_dict()
diag["tl_market_days_missing"] = stat[stat.status != "200"].groupby("market").size().to_dict()
diag["tl_rows_by_market_year"] = rows.assign(y=rows.date.str[:4]).groupby(["market", "y"]).size().unstack(0).fillna(0).astype(int).to_dict()
sess = [d for d in cal if pd.Timestamp("2018-01-02") <= d <= cal[-1]]
TL = np.zeros((N, NS), np.uint8); keyset = {k: n for n, k in enumerate(keys)}
rows["sym"] = rows.symbol.str.strip().str.upper(); rows = rows[rows.sym.isin(J)]
by = {m: {d: set(g.sym) for d, g in rows[rows.market == m].groupby("date")} for m in rows.market.unique()}
okset = set(zip(okday.date, okday.market)); dropped = 0
for m in ("NASDAQ",):
    cur = set()
    for d in sess:
        ds = d.strftime("%Y-%m-%d")
        if (ds, m) in okset: cur = by.get(m, {}).get(ds, set())
        if not cur: continue
        i = cal.get_loc(d); fk = d.strftime("%Y%m") + ("a" if d.day <= 15 else "b"); fi = keyset.get(fk)
        for s in cur:
            j = J[s]
            if fi is not None and ST[fi, j] == 2: dropped += 1; continue
            TL[i, j] = 1
diag["tl_rows_dropped_identity"] = dropped; np.save(CORE + "/tl.npy", TL)

# ---- features per event
avail_np = np.array(avail, dtype="datetime64[ns]")
out = {k: np.full(len(E), np.nan) for k in ["tl_on", "tl_run", "tl_days60", "tl_days250", "tl_new5", "tl_off5", "ftd_last_adv", "ftd_max20_adv", "ftd_mean20_adv",
                                             "ftd_max60_adv", "ftd_days20", "ftd_days60", "ftd_run10k", "ftd_max20_tso", "ftd_thr_run", "ftd_dollar20", "ftd_chg", "ftd_stale"]}
def runlen(x):
    n = 0
    for v in x[::-1]:
        if v: n += 1
        else: break
    return n
for k, (j, i, t0) in enumerate(zip(E.j.values, E.i.values, E.t0.values)):
    j, i = int(j), int(i); col = TL[:, j]
    out["tl_on"][k] = col[i]; out["tl_run"][k] = runlen(col[max(0, i - 300):i + 1] == 1)
    out["tl_days60"][k] = col[i - 59:i + 1].sum(); out["tl_days250"][k] = col[max(0, i - 249):i + 1].sum()
    out["tl_new5"][k] = float(any(col[q] == 1 and col[q - 20:q].sum() == 0 for q in range(i - 4, i + 1)))
    out["tl_off5"][k] = float(any(col[q] == 0 and col[q - 1] == 1 and runlen(col[max(0, q - 300):q] == 1) >= 5 for q in range(i - 4, i + 1)))
    # FTD: largest prefix of files all usable at t0
    u = avail_np <= t0
    if not u[0]: continue
    fl = int(np.argmin(u)) - 1 if not u.all() else NF - 1
    if ST[fl, j] in (2, 3): continue
    h = int(FHI[fl]); w20, w60 = slice(max(0, h - 19), h + 1), slice(max(0, h - 59), h + 1)
    adv = np.nanmean(V[max(0, i - 20):i, j]); adv = adv if adv > 0 else np.nan
    fa20, fa60, fr20 = FA[w20, j].astype(float), FA[w60, j].astype(float), FR[w20, j].astype(float)
    out["ftd_last_adv"][k] = FA[h, j] / adv
    out["ftd_max20_adv"][k] = np.nanmax(fa20) / adv if np.isfinite(fa20).any() else np.nan
    out["ftd_mean20_adv"][k] = np.nanmean(fa20) / adv if np.isfinite(fa20).any() else np.nan
    out["ftd_max60_adv"][k] = np.nanmax(fa60) / adv if np.isfinite(fa60).any() else np.nan
    out["ftd_days20"][k] = np.sum(fa20 > 0); out["ftd_days60"][k] = np.sum(fa60 > 0)
    out["ftd_run10k"][k] = runlen(FR[max(0, h - 99):h + 1, j] >= 10000)
    tk = tso[k] * CORR[fl, j]   # same split correction as the fails near t0
    if np.isfinite(tk) and tk > 0:
        out["ftd_max20_tso"][k] = np.nanmax(fa20) / tk if np.isfinite(fa20).any() else np.nan
        seg = slice(max(0, h - 99), h + 1); out["ftd_thr_run"][k] = runlen((FR[seg, j] >= 10000) & (FA[seg, j] / tk >= 0.005))
    dv = FR[w20, j].astype(float) * PX[w20, j].astype(float); out["ftd_dollar20"][k] = np.log10(1 + np.nanmax(dv)) if np.isfinite(dv).any() else np.nan
    a, b = FA[max(0, h - 9):h + 1, j].astype(float), FA[max(0, h - 19):max(0, h - 9), j].astype(float)
    out["ftd_chg"][k] = (np.nanmean(a) - np.nanmean(b)) / adv if np.isfinite(a).any() and np.isfinite(b).any() else np.nan
    out["ftd_stale"][k] = (pd.Timestamp(t0) - pd.Timestamp(SD[h])).days
F = pd.DataFrame(out, index=E.index)
exch = pd.read_csv(CORE + "/meta/meta.csv").set_index("symbol").exchangeName.reindex(E.symbol).to_numpy()
nq = np.isin(exch, ["NCM", "NMS", "NGM"]); F.loc[~nq, [c for c in F.columns if c.startswith("tl_")]] = np.nan
diag["tl_defined_events"] = float(nq.mean())
F.to_pickle(CORE + "/ftd_feat.pkl")
diag["events"] = int(len(F)); diag["ftd_nonnan"] = float(F.ftd_days20.notna().mean()); diag["ftd_any60"] = float((F.ftd_days60 > 0).mean())
diag["tso_nonnan"] = float(np.isfinite(tso).mean()); diag["tl_on_events"] = int(F.tl_on.sum()); diag["tl_ever250_events"] = int((F.tl_days250 > 0).sum())
diag["ftd_stale_days"] = F.ftd_stale.describe().round(1).to_dict()
json.dump(diag, open(CORE + "/ftd_diag.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in diag.items() if k != "tl_rows_by_market_year"}, ensure_ascii=False, default=str, indent=1))
