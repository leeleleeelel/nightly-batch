# REBUILD 2026-11-18: unchanged copy, run on data/core.
# prereg_rel.json: relationship-dynamics features at t0 (data up to the t0 close; entry is the t0+1 open)
# 15 series -> 105 pairs (c10, jump, z10, zmax20, zlag), 210 lead-lag pairs (ll20, lljump), univariate explosions, structural break, literature features
import sys, time, numpy as np, pandas as pd
CORE = sys.argv[1]; t_start = time.time()
E = pd.read_pickle(CORE + "/events_x19.pkl")[["j", "i"]]
C = np.load(CORE + "/px/c.npy"); T, N = C.shape
W = 181; ii = E.i.to_numpy(); jj = E.j.to_numpy(); nE = len(E)
IDX = ii[:, None] + np.arange(-W + 1, 1)[None, :]; assert IDX.min() >= 0
def gather(A): return A[IDX, jj[:, None]].astype(np.float32)
def df(A): return pd.DataFrame(A)
Cp = np.vstack([np.full((1, N), np.nan, np.float32), C[:-1]])
S = {}
with np.errstate(all="ignore"):
    O = np.load(CORE + "/px/o.npy"); H = np.load(CORE + "/px/h.npy"); L = np.load(CORE + "/px/l.npy"); V = np.load(CORE + "/px/v.npy")
    r = np.clip(C / Cp - 1, -0.95, 5.0); S["r"] = gather(r)
    S["dv"] = gather(np.where(C * V > 0, np.log(C * V), np.nan))
    S["rng"] = gather((H - L) / Cp)
    S["gap"] = gather(np.clip(O / Cp - 1, -0.95, 5.0))
    S["intr"] = gather(np.clip(C / O - 1, -0.95, 5.0))
    S["clv"] = gather(np.where(H > L, ((C - L) - (H - C)) / (H - L), np.nan))
    d = C - Cp; g = np.where(np.isfinite(d), np.maximum(d, 0), np.nan); lo = np.where(np.isfinite(d), np.maximum(-d, 0), np.nan)
    ag = df(g).ewm(alpha=1 / 14, adjust=False, ignore_na=True, min_periods=14).mean().to_numpy(); al = df(lo).ewm(alpha=1 / 14, adjust=False, ignore_na=True, min_periods=14).mean().to_numpy()
    S["rsi"] = gather(np.where(al > 0, 100 - 100 / (1 + ag / al), np.where(np.isfinite(al), 100.0, np.nan))); del d, g, lo, ag, al
    e12 = df(C).ewm(span=12, adjust=False, ignore_na=True).mean().to_numpy(); e26 = df(C).ewm(span=26, adjust=False, ignore_na=True).mean().to_numpy()
    macd = (e12 - e26) / C; sig = df(macd).ewm(span=9, adjust=False, ignore_na=True).mean().to_numpy()
    S["macd"] = gather(macd); S["mhist"] = gather(macd - sig); del e12, e26, macd, sig
    sma = df(C).rolling(20, min_periods=15).mean().to_numpy(); sd = df(C).rolling(20, min_periods=15).std().to_numpy()
    S["pb"] = gather(np.where(sd > 0, (C - (sma - 2 * sd)) / (4 * sd), np.nan)); del sma, sd
    v20 = df(V).rolling(20, min_periods=10).mean().shift(1).to_numpy(); vz = np.log((V + 1) / (v20 + 1)); S["vz"] = gather(vz)
    S["svr"] = gather(np.load(CORE + "/svr.npy"))
    ll = df(L).rolling(14, min_periods=10).min().to_numpy(); hh = df(H).rolling(14, min_periods=10).max().to_numpy()
    S["stk"] = gather(np.where(hh > ll, (C - ll) / (hh - ll), np.nan)); del ll, hh
    mk = np.nanmean(np.clip(r, -0.5, 1.0), axis=1).astype(np.float32); S["mkt"] = mk[IDX].astype(np.float32)
    S["sv"] = gather(np.sign(r) * np.minimum(np.exp(vz), 50)); del O, H, L, V, r, vz, v20, Cp
del C
names = list(S); K = len(names); print("series", names, round(time.time() - t_start), "s", flush=True)
# demean each event row (numerical stability of rolling sums)
for k in names:
    S[k] = (S[k] - np.nanmean(S[k], axis=1, keepdims=True)).astype(np.float32)
def rs(a, w):
    c = np.cumsum(a, axis=1); out = np.full(a.shape, np.nan); out[:, w - 1] = c[:, w - 1]; out[:, w:] = c[:, w:] - c[:, :-w]; return out
def rcorr(x, y, w, minfrac=0.7):
    m = np.isfinite(x) & np.isfinite(y); x0 = np.where(m, x, 0.0).astype(np.float64); y0 = np.where(m, y, 0.0).astype(np.float64)
    n = rs(m.astype(np.float64), w); sx = rs(x0, w); sy = rs(y0, w)
    cov = n * rs(x0 * y0, w) - sx * sy; vx = n * rs(x0 * x0, w) - sx * sx; vy = n * rs(y0 * y0, w) - sy * sy
    with np.errstate(all="ignore"):
        c = cov / np.sqrt(vx * vy)
    c[(n < minfrac * w) | ~(vx > 1e-12 * n * n) | ~(vy > 1e-12 * n * n)] = np.nan
    return np.clip(c, -1, 1)
F = {}; Z10 = np.full((nE, K, K), np.nan, np.float32); C20 = np.full((nE, K, K), np.nan, np.float32); C60 = np.full((nE, K, K), np.nan, np.float32)
with np.errstate(all="ignore"):
    for a in range(K):
        for b in range(a + 1, K):
            x, y = S[names[a]], S[names[b]]; tag = f"{names[a]}~{names[b]}"
            r10 = rcorr(x, y, 10); r20 = rcorr(x, y, 20); r60 = rcorr(x, y, 60)
            c10, base60 = r10[:, 180], r60[:, 160]
            bl = r10[:, 40:161]; okn = np.isfinite(bl).sum(1) >= 60
            mu = np.where(okn, np.nanmean(bl, 1), np.nan); sd = np.maximum(np.where(okn, np.nanstd(bl, 1), np.nan), 0.05)
            z10 = (c10 - mu) / sd; zs = (r10[:, 161:181] - mu[:, None]) / sd[:, None]
            az = np.where(np.isfinite(zs), np.abs(zs), -1.0); am = az.argmax(1); zmax = zs[np.arange(nE), am]; zmax[az.max(1) < 0] = np.nan
            F[f"c10|{tag}"] = c10; F[f"jump|{tag}"] = c10 - base60; F[f"z10|{tag}"] = z10; F[f"zmax20|{tag}"] = zmax
            F[f"zlag|{tag}"] = np.where(np.isfinite(zmax), 19 - am, np.nan)
            Z10[:, a, b] = Z10[:, b, a] = z10; C20[:, a, b] = C20[:, b, a] = r20[:, 180]; C60[:, a, b] = C60[:, b, a] = base60
        print("pairs done for", names[a], round(time.time() - t_start), "s", flush=True)
    for a in range(K):
        xs = np.full_like(S[names[a]], np.nan); xs[:, 1:] = S[names[a]][:, :-1]
        for b in range(K):
            if a == b: continue
            tag = f"{names[a]}>{names[b]}"; l20 = rcorr(xs, S[names[b]], 20)[:, 180]; l60 = rcorr(xs, S[names[b]], 60)[:, 160]
            F[f"ll20|{tag}"] = l20; F[f"lljump|{tag}"] = l20 - l60
    print("lead-lag done", round(time.time() - t_start), "s", flush=True)
    for k in names:
        s = S[k].astype(np.float64); bl = s[:, 40:160]; ok = np.isfinite(bl).sum(1) >= 60
        mu, sd = np.nanmean(bl, 1), np.nanstd(bl, 1); sd = np.where(ok & (sd > 0), sd, np.nan)
        F[f"uz1|{k}"] = (s[:, 180] - mu) / sd
        m5 = pd.DataFrame(s.T).rolling(5, min_periods=4).mean().to_numpy().T
        b5 = m5[:, 44:160]; F[f"uz5|{k}"] = (m5[:, 180] - np.nanmean(b5, 1)) / np.where(np.nanstd(b5, 1) > 0, np.nanstd(b5, 1), np.nan)
        d2 = np.full_like(m5, np.nan); d2[:, 10:] = (m5[:, 10:] - m5[:, 5:-5]) - (m5[:, 5:-5] - m5[:, :-10])
        bd = d2[:, 54:160]; sdd = np.nanstd(bd, 1); z2 = (d2[:, 180] - np.nanmean(bd, 1)) / np.where(sdd > 0, sdd, np.nan)
        F[f"ud2z|{k}"] = z2; prev = m5[:, 175] - m5[:, 170]
        F[f"uflip|{k}"] = np.where(np.isfinite(z2) & np.isfinite(prev), ((np.sign(d2[:, 180]) == -np.sign(prev)) & (np.abs(z2) >= 1)).astype(float), np.nan)
    az = np.abs(Z10)
    F["sb_n_z2"] = np.nansum(az >= 2, axis=(1, 2)) / 2; F["sb_n_z3"] = np.nansum(az >= 3, axis=(1, 2)) / 2
    zm = np.stack([np.abs(F[f"zmax20|{names[a]}~{names[b]}"]) for a in range(K) for b in range(a + 1, K)], 1)
    F["sb_n_zmax3"] = np.nansum(zm >= 3, 1); F["sb_max_absz"] = np.nanmax(np.where(np.isfinite(az), az, -1), axis=(1, 2)); F["sb_max_absz"][F["sb_max_absz"] < 0] = np.nan
    F["sb_mean_absz"] = np.nanmean(az.reshape(nE, -1), 1)
    cnt = np.nansum(az >= 2, axis=2); hub = cnt.argmax(1); has = cnt.max(1) > 0
    for k in range(K): F[f"sb_hub|{names[k]}"] = ((hub == k) & has).astype(float)
    F["sb_n_uz2"] = np.sum(np.stack([np.abs(F[f"uz1|{k}"]) >= 2 for k in names], 1), 1).astype(float)
    dfr = C20 - C60; okp = np.isfinite(dfr).sum((1, 2)) >= 0.7 * K * (K - 1)
    F["sb_frob"] = np.where(okp, np.sqrt(np.nansum(dfr ** 2, axis=(1, 2)) / np.maximum(np.isfinite(dfr).sum((1, 2)), 1)), np.nan)
    for nm, M in (("20", C20), ("60", C60)):
        M = np.where(np.isfinite(M), M, 0.0); M[:, np.arange(K), np.arange(K)] = 1.0
        ev = np.linalg.eigvalsh(M.astype(np.float64)); F[f"sb_eig1_{nm}"] = np.where(okp, ev[:, -1] / K, np.nan)
    F["sb_d_eig1"] = F["sb_eig1_20"] - F["sb_eig1_60"]
    # literature features (raw, not demeaned, series re-gathered where the level matters)
    Cx = np.load(CORE + "/px/c.npy"); Cpx = np.vstack([np.full((1, N), np.nan, np.float32), Cx[:-1]]); rr = gather(np.clip(Cx / Cpx - 1, -0.95, 5.0)).astype(np.float64); del Cx, Cpx
    Vx = np.load(CORE + "/px/v.npy"); v20 = df(Vx).rolling(20, min_periods=10).mean().shift(1).to_numpy(); vzr = gather(np.log((Vx + 1) / (v20 + 1))).astype(np.float64); del Vx, v20
    for nm, a0 in (("fip20", 158), ("fip60", 118)):
        seg = rr[:, a0:178]; cum = np.nanprod(1 + seg, 1) - 1; fin = np.isfinite(seg).sum(1)
        F[nm] = np.where(fin >= 0.7 * seg.shape[1], np.sign(cum) * ((seg < 0).sum(1) - (seg > 0).sum(1)) / np.maximum(fin, 1), np.nan)
    pos = np.clip(rr[:, 178:181], 0, None); sp = np.nansum(pos, 1); F["surge_conc"] = np.where(sp > 0, np.nanmax(pos, 1) / sp, np.nan)
    F["maxret21"] = np.nanmax(rr[:, 157:178], 1); F["hvp"] = np.nanmean(vzr[:, 173:178], 1)
    y = rr[:, 60:178]; x1 = rr[:, 59:177]; x2 = x1 * vzr[:, 59:177]; m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    X = np.stack([np.ones_like(x1), np.where(m, x1, 0), np.where(m, x2, 0)], 2) * m[:, :, None]; Y = np.where(m, y, 0)
    XtX = np.einsum("nti,ntj->nij", X, X) + np.eye(3) * 1e-9; Xty = np.einsum("nti,nt->ni", X, Y)
    beta = np.linalg.solve(XtX, Xty[:, :, None])[:, :, 0]; F["llor_c2"] = np.where(m.sum(1) >= 60, beta[:, 2], np.nan)
R = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()}, index=E.index).replace([np.inf, -np.inf], np.nan)
R.to_pickle(CORE + "/rel_feat.pkl")
cov = R.notna().mean(); print("features", R.shape, "| coverage median", round(float(cov.median()), 3), "min", round(float(cov.min()), 3), "| low-coverage:", list(cov[cov < 0.5].index[:8]))
print("examples:", R[["z10|dv~rsi", "z10|rsi~macd", "sb_n_z2", "sb_frob", "fip20", "llor_c2"]].describe().round(3).to_string())
print("done", round(time.time() - t_start), "s")
