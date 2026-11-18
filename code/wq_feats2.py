# REBUILD 2026-11-18: unchanged copy, run on data/core.
# BRAIN-style operator features for every surge event, evaluated at T = t0 + OFF (OFF=0: decision at the t0 close; OFF=2: at the t0+2 close).
# Fields per stock-day (log volume, log return, range, gap, intraday, close location, wick, VWAP-proxy deviation, log dollar volume, turnover,
# FINRA short-volume ratio, short interest/shares, return minus market, |return|, market and SIC-2 group returns). Operators: ts_mean/std/zscore/
# rank/arg_max/arg_min/skewness/kurtosis/delta/scale/decay/slope/share-positive, acceleration, ts_corr + correlation jumps + beta, n-th differences
# with sign flips against the prior trend, behaviour after the largest volume day, cross-sectional rank (all stocks that day), group-relative.
# Usage: wq_feats.py CORE OFF   -> CORE/wq_T{OFF}.pkl
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
CORE, OFF = sys.argv[1], int(sys.argv[2])
E = pd.read_pickle(CORE + "/events_x19.pkl")
cal = pd.DatetimeIndex(np.load(CORE + "/px/cal.npy")); N = len(cal); syms = json.load(open(CORE + "/px/syms.json")); S_ = len(syms)
A = {c: np.ascontiguousarray(np.load(CORE + f"/px/{c}.npy").T) for c in "ohlcv"}          # [symbol, session]
SV = np.ascontiguousarray(np.load(CORE + "/svr.npy").T)
with np.errstate(divide="ignore", invalid="ignore"):
    LR = np.full((S_, N), np.nan, np.float32); LR[:, 1:] = np.log(A["c"][:, 1:] / A["c"][:, :-1])
LR[~np.isfinite(LR)] = np.nan; LR = np.clip(LR, -1.0, 1.0)
mret = np.nanmedian(np.clip(LR, -0.5, 0.5), axis=0)                                       # market median log return per session
comp = pd.read_csv(CORE + "/meta/sec_company.csv", dtype=str, keep_default_na=False); mp = pd.read_csv(CORE + "/meta/cik_map.csv", dtype=str, keep_default_na=False)
sic2 = pd.Series(mp.cik.map(dict(zip(comp.cik, comp.sic))).fillna("").str[:2].values, index=mp.symbol).to_dict()
grp = np.array([sic2.get(s, "") for s in syms]); gid = {g: k for k, g in enumerate([g for g in sorted(set(grp)) if g and (grp == g).sum() >= 5])}
GM = np.full((len(gid), N), np.nan, np.float32)
for g, k in gid.items(): GM[k] = np.nanmedian(np.clip(LR[grp == g], -0.5, 0.5), axis=0)
# shares (raw) and split factor -> turnover; short interest / shares
spl = pd.read_csv(CORE + "/px/splits_x19.csv"); spl["date"] = pd.to_datetime(spl.date); spl["f"] = spl.numerator / spl.denominator
X = pd.read_pickle(CORE + "/xbrl_facts.pkl"); X = X[X.tag == "EntityCommonStockSharesOutstanding"]
cik_of = {s: int(c) for s, c in zip(mp.symbol, mp.cik) if c}
shx = {c: g.sort_values("avail") for c, g in X.groupby("cik")}
si = pd.read_pickle(CORE + "/si.pkl"); si["avail"] = si.settle + pd.Timedelta(days=14); six = {s: g.sort_values("avail") for s, g in si.groupby("symbol")}
need = sorted(set(E.j))
TOV = {}; SIP = {}
for j in need:
    s = syms[j]; fut = np.ones(N)
    for d0, f in zip(spl.date[spl.symbol == s], spl.f[spl.symbol == s]): fut[: cal.searchsorted(d0)] *= f
    sh = np.full(N, np.nan); g = shx.get(cik_of.get(s, -1))
    if g is not None:
        for av, v in zip(g.avail, g.val):
            k = cal.searchsorted(av)
            if k < N: sh[k:] = v
    with np.errstate(divide="ignore", invalid="ignore"): TOV[j] = (A["v"][j] / (sh / fut)).astype(np.float32)
    sp = np.full(N, np.nan); g = six.get(s)
    if g is not None:
        for av, v in zip(g.avail, g.si):
            k = cal.searchsorted(av)
            if k < N: sp[k:] = v
    with np.errstate(divide="ignore", invalid="ignore"): SIP[j] = (sp / sh).astype(np.float32)
# ---------- event windows [events, W]: sessions t0-261 .. t0+2 ----------
B, W = 261, 264
J = E.j.to_numpy(); I = E.i.to_numpy(); idx = I[:, None] + np.arange(-B, W - B)[None, :]
ok = idx < N; idxc = np.minimum(idx, N - 1)
def win(M):
    out = M[J[:, None], idxc].astype(np.float32); out[~ok] = np.nan; return out
o, h, l, c, v = (win(A[k]) for k in "ohlcv")
with np.errstate(divide="ignore", invalid="ignore"):
    F = {"logv": np.log1p(v), "ret": win(LR), "rng": (h - l) / c, "gap": np.log(o[:, 1:] / c[:, :-1]), "intr": np.log(c / o), "clv": np.where(h > l, (c - l) / (h - l), np.nan),
         "wick": np.where(h > l, (h - np.maximum(o, c)) / (h - l), np.nan), "vwapdev": np.log(c / ((h + l + c) / 3)), "logdv": np.log1p(c * v), "svr": win(SV),
         "tov": np.log(np.stack([TOV[j] for j in J])[np.arange(len(J))[:, None], idxc] + 1e-6), "sipct": np.stack([SIP[j] for j in J])[np.arange(len(J))[:, None], idxc]}
F["gap"] = np.c_[np.full((len(J), 1), np.nan, np.float32), F["gap"]]
F["mret"] = mret[idxc].astype(np.float32); F["rel"] = F["ret"] - F["mret"]; F["absret"] = np.abs(F["ret"])
gi = np.array([gid.get(grp[j], -1) for j in J]); F["sret"] = np.where(gi[:, None] >= 0, GM[np.maximum(gi, 0)[:, None], idxc], np.nan).astype(np.float32)
for k in F: F[k] = np.where(np.isfinite(F[k]), F[k], np.nan).astype(np.float32); F[k][~ok] = np.nan
del o, h, l, v
T = B + OFF                                                                                 # column of the evaluation day
out = {}
def sl(x, d, end=T): return x[:, end - d + 1:end + 1]
def nstd(s): return np.nanstd(s, axis=1, ddof=1)
RET_LIKE = {"ret", "gap", "intr", "rel", "vwapdev"}
for f in ["logv", "ret", "rng", "gap", "intr", "clv", "wick", "vwapdev", "logdv", "tov", "svr", "sipct", "rel", "absret"]:
    x = F[f]; xT = x[:, T]
    for d in (5, 10, 20, 60):
        s = sl(x, d); m = np.nanmean(s, 1); sd = nstd(s); n = np.isfinite(s).sum(1); good = n >= max(3, d // 2)
        put = lambda k, val: out.__setitem__(f"{f}|{k}{d}", np.where(good, val, np.nan))
        put("mean", m); put("std", sd); put("z", (xT - m) / sd); put("rank", (s < xT[:, None]).sum(1) / np.maximum(n, 1))
        sf = np.where(np.isfinite(s), s, -np.inf); put("argmax", d - 1 - np.argmax(sf, 1)); sf = np.where(np.isfinite(s), s, np.inf); put("argmin", d - 1 - np.argmin(sf, 1))
        mx, mn = np.nanmax(s, 1), np.nanmin(s, 1); put("scale", (xT - mn) / (mx - mn)); put("delta", xT - x[:, T - d])
        w = np.arange(1, d + 1, dtype=np.float32)[None, :] * np.isfinite(s); put("decay", np.nansum(np.nan_to_num(s) * w, 1) / np.maximum(w.sum(1), 1e-9) - m)
        t = np.arange(d, dtype=np.float32)[None, :]; fm = np.isfinite(s); tb = (t * fm).sum(1) / np.maximum(fm.sum(1), 1)
        cov = np.nansum((t - tb[:, None]) * (s - m[:, None]), 1); vt = np.nansum(((t - tb[:, None]) ** 2) * fm, 1); put("slope", cov / np.maximum(vt, 1e-9) * d / sd)
        if d >= 20:
            zz = (s - m[:, None]) / sd[:, None]; put("skew", np.nanmean(zz ** 3, 1)); put("kurt", np.nanmean(zz ** 4, 1) - 3)
        if f in RET_LIKE: put("pos", np.nansum(s > 0, 1) / np.maximum(n, 1))
    m5 = pd.DataFrame(x.T).rolling(5, min_periods=3).mean().to_numpy().T
    out[f"{f}|accel1"] = m5[:, T] - m5[:, T - 5]; out[f"{f}|accel2"] = m5[:, T] - 2 * m5[:, T - 5] + m5[:, T - 10]
# correlations, correlation jumps, beta
def ncorr(a, b):
    mk = np.isfinite(a) & np.isfinite(b); n = mk.sum(1); a0 = np.where(mk, a, 0); b0 = np.where(mk, b, 0)
    ma, mb = a0.sum(1) / np.maximum(n, 1), b0.sum(1) / np.maximum(n, 1); da, db = np.where(mk, a - ma[:, None], 0), np.where(mk, b - mb[:, None], 0)
    r = (da * db).sum(1) / np.sqrt(np.maximum((da ** 2).sum(1) * (db ** 2).sum(1), 1e-18)); return np.where(n >= 6, r, np.nan)
PAIRS = [("ret", "logv"), ("ret", "svr"), ("logv", "svr"), ("ret", "mret"), ("ret", "sret"), ("rng", "logv"), ("gap", "logv"), ("intr", "gap"), ("absret", "logv"),
         ("clv", "logv"), ("ret", "ret_l1"), ("logv", "logv_l1")]
F["ret_l1"] = np.c_[np.full((len(J), 1), np.nan, np.float32), F["ret"][:, :-1]]; F["logv_l1"] = np.c_[np.full((len(J), 1), np.nan, np.float32), F["logv"][:, :-1]]
for a, b in PAIRS:
    cs = {d: ncorr(sl(F[a], d), sl(F[b], d)) for d in (10, 20, 60)}
    for d in (10, 20, 60): out[f"corr({a},{b})|{d}"] = cs[d]
    out[f"corr({a},{b})|jump10_60"] = cs[10] - cs[60]; out[f"corr({a},{b})|jump20_60"] = cs[20] - cs[60]
for d in (20, 60):
    a, b = sl(F["ret"], d), sl(F["mret"], d); mk = np.isfinite(a) & np.isfinite(b)
    bm = np.where(mk, b, np.nan); cov = np.nanmean((a - np.nanmean(np.where(mk, a, np.nan), 1)[:, None]) * (bm - np.nanmean(bm, 1)[:, None]), 1)
    out[f"beta|{d}"] = cov / np.maximum(np.nanvar(bm, 1), 1e-12)
out["beta|jump"] = out["beta|20"] - out["beta|60"]
# n-th differences of smoothed series: magnitude, sign flip against the prior 5-day direction, sign changes in 20 days
for f in ["logv", "svr", "rng", "clv", "logdv"]:
    s3 = pd.DataFrame(F[f].T).rolling(3, min_periods=2).mean().to_numpy().T; sd60 = nstd(sl(F[f], 60))
    D = s3
    for n_ in (1, 2, 3):
        D = np.c_[np.full((len(J), 1), np.nan), np.diff(D, axis=1)]
        dT = D[:, T]; prev = np.nanmean(D[:, T - 5:T], 1); sgn = np.sign(dT); flip = (sgn != 0) & (np.sign(prev) == -sgn)
        out[f"{f}|d{n_}"] = dT / sd60; out[f"{f}|d{n_}flip"] = np.where(np.isfinite(dT) & np.isfinite(prev), flip * sgn, np.nan)
        z20 = np.sign(D[:, T - 19:T + 1]); out[f"{f}|d{n_}chg20"] = np.nansum(np.abs(np.diff(z20, axis=1)) > 0, 1).astype(float)
# behaviour after the largest volume day in the previous L sessions (excluding T)
pos = np.arange(W)[None, :]
for L in (20, 60):
    s = sl(F["logv"], L, T - 1); sf = np.where(np.isfinite(s), s, -np.inf); kst = (T - L) + np.argmax(sf, 1); valid = np.isfinite(s).sum(1) >= L // 2
    out[f"vmaxday{L}|since"] = np.where(valid, T - kst, np.nan)
    aft = (pos > kst[:, None]) & (pos <= T); bef = (pos >= kst[:, None] - 20) & (pos < kst[:, None]); enough = (T - kst) >= 3
    for f in ["ret", "rng", "svr", "gap", "clv", "rel", "absret"]:
        x = F[f]; xa, xb = np.where(aft, x, np.nan), np.where(bef, x, np.nan)
        out[f"vmaxday{L}|{f}_volratio"] = np.where(valid & enough, np.log(np.nanstd(xa, 1) / np.nanstd(xb, 1)), np.nan)
        out[f"vmaxday{L}|{f}_meandiff"] = np.where(valid & enough, np.nanmean(xa, 1) - np.nanmean(xb, 1), np.nan)
    cT = c[:, T]; ck = c[np.arange(len(J)), kst]; out[f"vmaxday{L}|drift"] = np.where(valid, np.log(cT / ck), np.nan)
    out[f"vmaxday{L}|vdecay"] = np.where(valid, F["logv"][:, T] - F["logv"][np.arange(len(J)), kst], np.nan)
# cross-sectional rank among all stocks on day T, and SIC-2 group-relative values
dT_idx = np.minimum(I + OFF, N - 1)
def xs_rank(Q, vals):
    r = np.full(len(J), np.nan)
    for d in np.unique(dT_idx):
        row = Q[:, d]; row = np.sort(row[np.isfinite(row)]); e = np.flatnonzero(dT_idx == d)
        if len(row) >= 100: r[e] = np.searchsorted(row, vals[e]) / len(row)
    return r
CUM = np.nancumsum(np.nan_to_num(LR), axis=1)
def qmat(k):
    with np.errstate(divide="ignore", invalid="ignore"):
        if k == "r1": return LR
        if k in ("r5", "r20", "r60"):
            dd = int(k[1:]); q = np.full_like(CUM, np.nan); q[:, dd:] = CUM[:, dd:] - CUM[:, :-dd]; return q
        if k == "rng": return (A["h"] - A["l"]) / A["c"]
        if k == "vratio":
            LV = np.log1p(A["v"]); return LV - pd.DataFrame(LV.T).rolling(60, min_periods=40).median().shift(1).to_numpy().T
        if k == "svr": return SV
        if k == "dv20": return pd.DataFrame(np.log1p(A["c"] * A["v"]).T).rolling(20, min_periods=15).median().to_numpy().T
        if k == "vol20": return pd.DataFrame(LR.T).rolling(20, min_periods=15).std().to_numpy().T
        if k == "gap": return np.c_[np.full((S_, 1), np.nan), np.log(A["o"][:, 1:] / A["c"][:, :-1])]
for k in ["r1", "r5", "r20", "r60", "rng", "vratio", "svr", "dv20", "vol20", "gap"]:
    Qm = qmat(k); Qm = np.where(np.isfinite(Qm), Qm, np.nan).astype(np.float32); vals = Qm[J, dT_idx]
    out[f"xsrank|{k}"] = xs_rank(Qm, vals)
    if k in ("r1", "r20", "svr"):
        gmed = np.full(len(J), np.nan)
        for g, kk in gid.items():
            e = np.flatnonzero(grp[J] == g)
            if len(e): gmed[e] = np.nanmedian(Qm[grp == g][:, dT_idx[e]], axis=0)
        out[f"grouprel|{k}"] = vals - gmed
    del Qm
O = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, index=E.index)
O = O.replace([np.inf, -np.inf], np.nan); O.to_pickle(CORE + f"/wq_T{OFF}.pkl")
print("features", O.shape[1], "events", len(O), "| mean non-missing", round(float(O.notna().mean().mean()), 3))
print(O.notna().mean().sort_values().head(8).round(2).to_dict())
