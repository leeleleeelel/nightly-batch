# REBUILD 2026-11-18: regime.py up to the market-state table (no regime clustering; 'regime' kept as a placeholder column, dropped by stage 2).
# Part R (prereg_w.json): data-defined market regimes and a regime-aware selection model (attempt 14).
# Daily market state from the whole universe; k-means on 2019-2023 days (k by silhouette 2..8); later days -> nearest centroid.
import sys, json, datetime, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
CORE = sys.argv[1]; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from detect import triggers
cal = pd.DatetimeIndex(np.load(CORE + "/px/cal.npy")); N = len(cal); syms = json.load(open(CORE + "/px/syms.json"))
A = {c: np.ascontiguousarray(np.load(CORE + f"/px/{c}.npy").T) for c in "ohlcv"}; SV = np.ascontiguousarray(np.load(CORE + "/svr.npy").T)
C = A["c"].astype(np.float64)
with np.errstate(divide="ignore", invalid="ignore"):
    LR = np.full(C.shape, np.nan); LR[:, 1:] = np.log(C[:, 1:] / C[:, :-1]); LR[~np.isfinite(LR)] = np.nan; LR = np.clip(LR, -1, 1)
    DV = pd.DataFrame((C * A["v"]).T).rolling(20, min_periods=15).median().to_numpy().T
elig = np.isfinite(C) & (DV >= 250000)
CUM = np.nancumsum(np.nan_to_num(LR), axis=1)
def rs(d): q = np.full(CUM.shape, np.nan); q[:, d:] = CUM[:, d:] - CUM[:, :-d]; return np.where(elig, q, np.nan)
r20, r60 = rs(20), rs(60)
ma50 = pd.DataFrame(C.T).rolling(50, min_periods=40).mean().to_numpy().T; hi250 = pd.DataFrame(C.T).rolling(250, min_periods=120).max().to_numpy().T
med = np.nanmedian(np.where(elig, np.clip(LR, -0.5, 0.5), np.nan), axis=0)
ter = np.nanpercentile(np.where(elig, DV, np.nan), [33.3, 66.7], axis=0)
small = np.where(elig & (DV <= ter[0]), r20, np.nan); large = np.where(elig & (DV >= ter[1]), r20, np.nan)
svr20 = pd.DataFrame(SV.T).rolling(20, min_periods=10).mean().to_numpy().T
trig = np.zeros(N)
for j in range(len(syms)):
    O, H, L, Cc, V = (A[k][j].astype(float) for k in "ohlcv"); a = np.c_[O, H, L, Cc, V]
    valid = np.isfinite(a).all(axis=1) & (O > 0) & (H > 0) & (L > 0) & (Cc > 0) & (V >= 0) & (H >= np.maximum(O, Cc) * (1 - 1e-5)) & (L <= np.minimum(O, Cc) * (1 + 1e-5)) & (H >= L)
    if valid.sum() < 80: continue
    for i in triggers(O, H, L, Cc, V, valid, cal, pd.Timestamp("2018-03-01"), cal[-1]): trig[i] += 1
E = pd.read_pickle(CORE + "/events_x19.pkl")
lab = E[E.labeled == True]; ge_by = lab.groupby("i").y3x.apply(lambda x: (x == True).sum()); n_by = lab.groupby("i").size()
gsum, nsum = np.zeros(N), np.zeros(N); gsum[ge_by.index] = ge_by.values; nsum[n_by.index] = n_by.values
G = pd.Series(gsum).rolling(65, min_periods=20).sum().shift(25).to_numpy(); Nn = pd.Series(nsum).rolling(65, min_periods=20).sum().shift(25).to_numpy()
with np.errstate(divide="ignore", invalid="ignore"):
    M = pd.DataFrame({"m_ret20": np.nanmedian(r20, axis=0), "m_ret60": np.nanmedian(r60, axis=0), "breadth50": np.nanmean(np.where(elig, C > ma50, np.nan), axis=0),
                      "disp20": np.nanpercentile(r20, 75, axis=0) - np.nanpercentile(r20, 25, axis=0), "mvol20": pd.Series(med).rolling(20, min_periods=15).std().to_numpy(),
                      "small_minus_large": np.nanmedian(small, axis=0) - np.nanmedian(large, axis=0),
                      "trig_rate20": pd.Series(trig).rolling(20, min_periods=15).sum().to_numpy() / np.maximum(elig.sum(axis=0), 1),
                      "svr_med20": np.nanmedian(np.where(elig, svr20, np.nan), axis=0), "high250_share": np.nanmean(np.where(elig, C >= hi250 * 0.999, np.nan), axis=0),
                      "ge_rate_known": G / np.maximum(Nn, 1)}, index=cal)
M["regime"] = -1; M.to_pickle(CORE + "/market_state.pkl"); print("market_state", M.shape, M.index.min().date(), M.index.max().date())
