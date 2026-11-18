# REBUILD 2026-11-18: feats_x19_b.py reading data/finra/{shvol,si}.csv.gz; trigger/label horizon = latest session. W = data/work, S = data.
# Decision-time features for ALL events 2019..2026 (same computation in every year), part B:
#  FINRA daily short-volume ratio (level, trend, spikes), FINRA short interest known at t0 (settle + 14 days),
#  daily price/volume path signatures before t0, the stock's own trigger / golden-egg history (raw triggers from 2018-03, 180-session look-back),
#  peer and market context (market-wide triggers, return-correlated peers, same-SIC peers and whether they hold), calendar position.
import sys, os, glob, json, numpy as np, pandas as pd
W, PX = sys.argv[1], sys.argv[2]; S = W[: -len("/work")] if W.endswith("/work") else W
sys.path.insert(0, W); sys.path.insert(0, W + "/x19")
from detect import triggers
from labeler import label
E = pd.read_pickle(W + "/x19/events_x19.pkl")
cal = pd.DatetimeIndex(np.load(PX + "/cal.npy")); N = len(cal); syms = json.load(open(PX + "/syms.json")); col = {s: j for j, s in enumerate(syms)}
A = {c: np.ascontiguousarray(np.load(PX + f"/{c}.npy").T) for c in "ohlcv"}
spl = pd.read_csv(PX + "/splits_x19.csv"); spl["date"] = pd.to_datetime(spl.date); splits = {s: g.date.tolist() for s, g in spl.groupby("symbol")}
def arrays(j):
    O, H, L, C, V = (A[c][j].astype(float) for c in "ohlcv")
    a = np.c_[O, H, L, C, V]
    valid = np.isfinite(a).all(axis=1) & (O > 0) & (H > 0) & (L > 0) & (C > 0) & (V >= 0) & (H >= np.maximum(O, C) * (1 - 1e-5)) & (L <= np.minimum(O, C) * (1 + 1e-5)) & (H >= L)
    return O, H, L, C, V, valid
pos = pd.Series(np.arange(N), index=cal)
out = {idx: {} for idx in E.index}
# ---------------- FINRA daily short-volume ratio matrix ----------------
SV = np.full((N, len(syms)), np.nan, dtype=np.float32)
x = pd.read_csv(S + "/finra/shvol.csv.gz", dtype={"Symbol": str}, keep_default_na=False)
x["d"] = pd.to_datetime(x.Date.astype(str)); parts = [x[["d", "Symbol", "ShortVolume", "TotalVolume"]]]
for p in sorted(glob.glob(S + "/finra_raw/CNMSshvol*.txt")):
    y = pd.read_csv(p, sep="|", dtype={"Symbol": str}, usecols=["Symbol", "ShortVolume", "TotalVolume"], keep_default_na=False); y["d"] = pd.Timestamp(os.path.basename(p)[9:17]); parts.append(y)
X = pd.concat(parts, ignore_index=True); X = X[X.d.isin(pos.index) & X.Symbol.isin(col)]
X["TotalVolume"] = pd.to_numeric(X.TotalVolume, errors="coerce"); X["ShortVolume"] = pd.to_numeric(X.ShortVolume, errors="coerce"); X = X[X.TotalVolume > 0]
SV[pos[X.d].to_numpy(), X.Symbol.map(col).to_numpy()] = (X.ShortVolume / X.TotalVolume).to_numpy(dtype=np.float32); del X, parts
for idx, e in E.iterrows():
    v = SV[:, e.j]; i = e.i; f = out[idx]
    a20, a5, b15 = v[i - 20:i], v[i - 5:i], v[i - 20:i - 5]
    if np.isfinite(a20).sum() >= 10:
        f.update(svr_t0=float(v[i]), svr_pre20=float(np.nanmean(a20)), svr_max20=float(np.nanmax(a20)), svr_std20=float(np.nanstd(a20)), svr_hi20=int(np.nansum(a20 >= 0.6)))
        f["svr_d"] = f["svr_t0"] - f["svr_pre20"]
        if np.isfinite(a5).sum() >= 3 and np.isfinite(b15).sum() >= 8: f.update(svr_pre5=float(np.nanmean(a5)), svr_trend=float(np.nanmean(a5) - np.nanmean(b15)))
        b60 = v[i - 60:i - 20]
        if np.isfinite(b60).sum() >= 20: f["svr_shift"] = float(np.nanmean(a20) - np.nanmean(b60))
# ---------------- FINRA short interest known at t0 ----------------
si = pd.read_csv(S + "/finra/si.csv.gz", dtype={"symbolCode": str}, keep_default_na=False)
for p in sorted(glob.glob(S + "/finra_si_raw/shrt*.csv")):
    y = pd.read_csv(p, sep="|", dtype={"symbolCode": str}, usecols=["symbolCode", "currentShortPositionQuantity", "daysToCoverQuantity", "changePercent", "averageDailyVolumeQuantity"], keep_default_na=False)
    y["settle"] = os.path.basename(p)[4:12]; si = pd.concat([si, y], ignore_index=True)
si = si.rename(columns={"symbolCode": "symbol", "currentShortPositionQuantity": "si", "daysToCoverQuantity": "dtc", "changePercent": "si_chg", "averageDailyVolumeQuantity": "adv"})
for c_ in ("si", "dtc", "si_chg", "adv"): si[c_] = pd.to_numeric(si[c_], errors="coerce")
si["settle"] = pd.to_datetime(si.settle.astype(str)); si = si.drop_duplicates(["symbol", "settle"]).sort_values(["symbol", "settle"])
si["si_prev2"] = si.groupby("symbol").si.shift(2); si["si_prev6"] = si.groupby("symbol").si.shift(6)
si["avail"] = si.settle + pd.Timedelta(days=14)
fa = pd.read_pickle(W + "/x19/feat_a.pkl")[["shares", "mcap"]]
Y = E[["symbol", "t0"]].join(fa).reset_index().sort_values("t0")
m = pd.merge_asof(Y, si.sort_values("avail"), left_on="t0", right_on="avail", by="symbol", direction="backward")
m = m[(m.t0 - m.avail).dt.days <= 45]
for r in m.itertuples():
    f = out[r.index]; f.update(si_shares=r.si, dtc=r.dtc, si_chg=r.si_chg, si_pct=r.si / r.shares if r.shares and r.shares == r.shares and r.shares > 0 else np.nan,
                               si_chg2=r.si / r.si_prev2 - 1 if r.si_prev2 and r.si_prev2 > 0 else np.nan, si_chg6=r.si / r.si_prev6 - 1 if r.si_prev6 and r.si_prev6 > 0 else np.nan)
# ---------------- raw triggers 2018-03.. for history / peers (all symbols, same trigger) ----------------
RT = []          # (symbol j, i, label y3x or nan, label known date)
R = np.full((N, len(syms)), np.nan, dtype=np.float32)
for j, s in enumerate(syms):
    O, H, L, C, V, valid = arrays(j)
    with np.errstate(divide="ignore", invalid="ignore"): R[1:, j] = np.log(C[1:] / C[:-1])
    if valid.sum() < 80: continue
    for i in triggers(O, H, L, C, V, valid, cal, pd.Timestamp("2018-03-01"), cal[-1]):
        lb = label(O, H, L, C, V, valid, cal, i, splits.get(s, []), cal[-1])
        known = cal[i - 3] + pd.Timedelta(days=32)
        RT.append((j, i, lb.get("y3x", np.nan) if lb["lab_valid"] else np.nan, known, lb.get("total_mult", np.nan), C[i]))
R[~np.isfinite(R)] = np.nan; R = np.clip(R, -1.0, 1.0)
RT = pd.DataFrame(RT, columns=["j", "i", "y", "known", "tmult", "c0"])
print("raw triggers", len(RT), "from", cal[RT.i.min()].date(), flush=True)
byj = {j: g.sort_values("i") for j, g in RT.groupby("j")}
byi = {i: g for i, g in RT.groupby("i")}
comp = pd.read_csv(W + "/sec_company.csv", dtype=str, keep_default_na=False); mp = pd.read_csv(S + "/sec_raw/cik_map.csv", dtype=str, keep_default_na=False)
c2sic = dict(zip(comp.cik, comp.sic)); sic_of = {col[s]: c2sic.get(c, "") for s, c in zip(mp.symbol, mp.cik) if s in col}
Cmat = A["c"]
med = np.nanmedian(R, axis=1); breadth5 = np.nanmean(R > np.log(1.05), axis=1)
cum20 = pd.DataFrame(R).rolling(20, min_periods=15).sum(); mkt20 = np.nanmedian(cum20.to_numpy(), axis=1); del cum20
ntrig = np.zeros(N); np.add.at(ntrig, RT.i.to_numpy(), 1)
active = np.isfinite(R).sum(axis=1)
trig_rate60 = pd.Series(ntrig).rolling(60, min_periods=60).sum().to_numpy() / np.maximum(pd.Series(active).rolling(60, min_periods=60).mean().to_numpy(), 1)
opex = []
for y in range(2018, 2027):
    for mo in range(1, 13):
        d = pd.Timestamp(y, mo, 15); third = d + pd.Timedelta(days=(4 - d.weekday()) % 7); k = cal.searchsorted(third, side="right") - 1
        if 0 <= k < N: opex.append(k)
opex = np.array(sorted(set(opex)))
mon = cal.to_period("M"); last_of_month = np.r_[mon[1:] != mon[:-1], True]
mend = np.zeros(N, int)
for k in range(N - 1, -1, -1): mend[k] = 0 if last_of_month[k] else mend[k + 1] + 1
for idx, e in E.iterrows():
    j, i, f = int(e.j), int(e.i), out[idx]
    O, H, L, C, V, valid = arrays(j)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.r_[np.nan, np.log(C[1:] / C[:-1])]
        w = slice(i - 20, i); rng = H[w] - L[w]
        clv = np.where(rng > 0, (C[w] - L[w]) / rng, np.nan); lw = np.where(rng > 0, (np.minimum(O[w], C[w]) - L[w]) / rng, np.nan)
        ovn = np.log(O[w] / C[i - 21:i - 1]); intra = np.log(C[w] / O[w])
        up = C[w] > C[i - 21:i - 1]; dn = C[w] < C[i - 21:i - 1]
        f.update(p_clv20=np.nanmean(clv), p_lowwick20=np.nanmean(lw), p_hiclose20=int(np.nansum(clv >= 0.9)), p_ovn20=np.nansum(ovn), p_intra20=np.nansum(intra),
                 p_ovn_share=np.nansum(np.abs(ovn)) / max(np.nansum(np.abs(ovn)) + np.nansum(np.abs(intra)), 1e-9),
                 p_rng_comp=np.nanmean(rng[-5:] / C[i - 5:i]) / max(np.nanmean(rng[:-5] / C[i - 20:i - 5]), 1e-9),
                 p_updown_vol=np.nansum(V[w][up]) / max(np.nansum(V[w][dn]), 1.0),
                 p_vspike20=np.nanmax(V[w]) / max(np.nanmedian(V[i - 80:i - 20]), 1.0),
                 p_amihud20=np.nanmean(np.abs(lr[w]) / np.maximum(C[w] * V[w], 1.0)) * 1e6, p_zero_vol60=int(np.nansum(V[i - 60:i] == 0)),
                 p_inside20=int(np.nansum((H[w] < H[i - 21:i - 1]) & (L[w] > L[i - 21:i - 1]))),
                 p_gapup20=int(np.nansum(ovn >= np.log(1.05))), p_gapdn20=int(np.nansum(ovn <= np.log(0.95))),
                 p_ma20=C[i - 1] / np.nanmean(C[i - 20:i]) - 1, p_ma50=C[i - 1] / np.nanmean(C[i - 50:i]) - 1,
                 p_lo250=C[i - 1] / np.nanmin(C[max(0, i - 250):i]) - 1, p_ret120=C[i - 1] / C[i - 121] - 1 if i >= 121 else np.nan,
                 p_ret250=C[i - 1] / C[i - 251] - 1 if i >= 251 and np.isfinite(C[i - 251]) else np.nan,
                 p_volratio=np.nanstd(lr[i - 20:i]) / max(np.nanstd(lr[i - 60:i]), 1e-9),
                 p_dvtrend=np.nanmedian((C * V)[i - 20:i]) / max(np.nanmedian((C * V)[i - 80:i - 20]), 1.0),
                 p_vol_rec=V[i] / max(np.nanmax(V[max(0, i - 250):i]), 1.0), p_r1_rec=lr[i] / max(np.nanmax(np.abs(lr[max(1, i - 250):i])), 1e-9))
        x60 = lr[i - 60:i]; x60 = x60[np.isfinite(x60)]
        if len(x60) >= 40:
            f["p_ac1"] = float(np.corrcoef(x60[:-1], x60[1:])[0, 1]); sd = x60.std()
            f["p_skew60"] = float(((x60 - x60.mean()) ** 3).mean() / sd ** 3) if sd > 0 else np.nan
        cc = C[i - 60:i]; f["p_maxdd60"] = float(np.nanmin(cc / np.fmax.accumulate(np.where(np.isfinite(cc), cc, 0))) - 1)
        k = 0
        while k < 30 and C[i - 1 - k] > C[i - 2 - k]: k += 1
        f["p_consec_up"] = k
        vr = np.nanmean(V[i - 5:i]) / max(np.nanmedian(V[i - 65:i - 5]), 1.0); pc = abs(C[i - 1] / C[i - 6] - 1)
        f.update(stealth_vol=vr, stealth_flat=float(vr >= 1.5 and pc <= 0.05), p_run180=np.nanmax(C[i - 180:i]) / max(np.nanmin(C[i - 180:i]), 1e-9))
    # own history (raw triggers in the previous 180 sessions)
    g = byj.get(j)
    if g is not None:
        h = g[(g.i < i) & (g.i >= i - 180)]
        f["own_trig180"] = len(h); f["own_since_last"] = float(i - h.i.max()) if len(h) else np.nan
        hk = h[h.known < e.t0]
        f["own_ge180"] = int((hk.y == True).sum()); f["own_maxmult180"] = float(hk.tmult.max()) if len(hk) else np.nan
    # peers and market
    a, b = i - 130, i - 10
    peers = [(jj, p, c0) for p in range(i - 10, i) if p in byi for jj, c0 in zip(byi[p].j, byi[p].c0) if jj != j]
    f["trig10"] = len(peers); f["trig_same_day"] = int(sum(1 for jj in (byi[i].j if i in byi else []) if jj != j))
    if peers and a >= 0:
        x = R[a:b, j]; Yp = R[a:b, [p[0] for p in peers]]
        mk = np.isfinite(x)[:, None] & np.isfinite(Yp); n = mk.sum(0)
        xm = np.where(mk, x[:, None], 0.0); ym = np.where(mk, Yp, 0.0)
        mx = xm.sum(0) / np.maximum(n, 1); my = ym.sum(0) / np.maximum(n, 1)
        dx = np.where(mk, x[:, None] - mx, 0.0); dy = np.where(mk, Yp - my, 0.0)
        cr = (dx * dy).sum(0) / np.sqrt(np.maximum((dx ** 2).sum(0) * (dy ** 2).sum(0), 1e-18)); cr = np.where(n >= 60, cr, np.nan)
        f.update(co10_03=int(np.nansum(cr >= 0.3)), co10_05=int(np.nansum(cr >= 0.5)), co10_max=float(np.nanmax(cr)) if np.isfinite(cr).any() else np.nan)
        hold = np.array([Cmat[jj][i - 1] >= c0 for jj, p, c0 in peers])
        sel = cr >= 0.3
        if sel.any(): f["co_peer_hold"] = int(np.sum(hold & sel))
        s4 = sic_of.get(j, "")
        if s4:
            same4 = np.array([sic_of.get(jj, "") == s4 for jj, p, c0 in peers]); same2 = np.array([sic_of.get(jj, "")[:2] == s4[:2] for jj, p, c0 in peers])
            f.update(sic4_peers10=int(same4.sum()), sic2_peers10=int(same2.sum()), sic4_hold10=int((same4 & hold).sum()), sic2_hold10=int((same2 & hold).sum()))
    f.update(mkt_median=float(med[i]), mkt_breadth5=float(breadth5[i]), mkt_ret20=float(mkt20[i - 1]), mkt_trig_rate60=float(trig_rate60[i - 1]))
    nxt = opex[opex >= i]; f["days_to_opex"] = int(nxt[0] - i) if len(nxt) else np.nan; f["opex_week"] = float(f["days_to_opex"] <= 4) if len(nxt) else np.nan
    f.update(weekday=int(cal[i].weekday()), month=int(cal[i].month), quad_month=float(cal[i].month in (3, 6, 9, 12)), days_to_month_end=int(mend[i]))
Xo = pd.DataFrame.from_dict(out, orient="index"); Xo.index.name = "idx"; Xo.to_pickle(W + "/x19/feat_b.pkl")
print(Xo.shape); print(Xo.notna().mean().round(2).sort_values().head(15).to_dict())
print(pd.concat([E.year, Xo[["svr_pre20", "si_pct", "dtc", "trig10", "co10_03", "own_trig180", "own_ge180", "p_clv20", "mkt_trig_rate60"]]], axis=1).groupby("year").mean().round(3).to_string())
