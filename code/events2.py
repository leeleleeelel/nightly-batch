# REBUILD 2026-11-18: events_x19.py with every event detected by the same trigger (no stored older events), 2019-01-02..latest session.
# One event table 2019-01..2026-11 on the unified panel: new events (t0 2019-01-02..2024-11-01, same trigger + pre-t0 guards as
# build_events.py) + the existing events (t0 >= 2024-11-02, identities and stored golden-egg labels kept). Decision-time price features are
# recomputed for ALL events from the same panel (full 250-day history everywhere), plus labels, and trend outcomes for entries at t0+1 / t0+2 open.
import sys, json, numpy as np, pandas as pd
PX, OUT = sys.argv[1], sys.argv[2]
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import triggers
from labeler import label
cal = pd.DatetimeIndex(np.load(PX + "/cal.npy")); N = len(cal); syms = json.load(open(PX + "/syms.json"))
A = {c: np.ascontiguousarray(np.load(PX + f"/{c}.npy").T) for c in "ohlcv"}   # [symbol, session]
spl = pd.read_csv(PX + "/splits_x19.csv"); spl["date"] = pd.to_datetime(spl.date); splits = {s: g.date.tolist() for s, g in spl.groupby("symbol")}
def arrays(j):
    O, H, L, C, V = (A[c][j].astype(float) for c in "ohlcv")
    a = np.c_[O, H, L, C, V]
    valid = np.isfinite(a).all(axis=1) & (O > 0) & (H > 0) & (L > 0) & (C > 0) & (V >= 0) & \
            (H >= np.maximum(O, C) * (1 - 1e-5)) & (L <= np.minimum(O, C) * (1 + 1e-5)) & (H >= L)
    return O, H, L, C, V, valid
old_by = {}
cpos = pd.Series(np.arange(N), index=cal)
LAST = cal[-1]
rows = []
for j, sym in enumerate(syms):
    O, H, L, C, V, valid = arrays(j)
    if valid.sum() < 80: continue
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.r_[np.nan, O[1:] / C[:-1]]; lr = np.r_[np.nan, np.log(C[1:] / C[:-1])]
    bad = np.isfinite(gap) & ((gap >= 4) | (gap <= 0.25)); ss = splits.get(sym, [])
    ev = []
    for i in triggers(O, H, L, C, V, valid, cal, pd.Timestamp("2019-01-02"), cal[-1]):
        d = cal[i]
        if bad[i - 60:i + 1].any() or any(d - pd.Timedelta(days=30) <= s <= d for s in ss): continue
        ev.append((i, "new", None, None))
    if sym in old_by:
        for idx, e in old_by[sym].iterrows(): ev.append((int(cpos[e.t0]), "old", e.y3x, bool(e.labeled)))
    for i, src, y_old, lab_old in ev:
        d = cal[i]; pv = np.median(V[i - 20:i]); rng = H[i] - L[i]; h250 = np.nanmax(C[max(0, i - 250):i])
        r = dict(symbol=sym, j=j, i=i, t0=d, src=src, price_pre=C[i - 1], t0_open=O[i], t0_close=C[i], r1=C[i] / C[i - 1] - 1, r3=C[i] / C[i - 3] - 1,
                 gap=O[i] / C[i - 1] - 1, intraday=C[i] / O[i] - 1, clv=(C[i] - L[i]) / rng if rng > 0 else np.nan,
                 upper_wick=(H[i] - max(O[i], C[i])) / rng if rng > 0 else np.nan, hi_vs_close=H[i] / C[i] - 1,
                 rvol=V[i] / pv if pv > 0 else np.nan, dv_t0=C[i] * V[i], dv20=np.median((C * V)[i - 20:i]),
                 pre5=C[i - 1] / C[i - 6] - 1, pre20=C[i - 1] / C[i - 21] - 1, pre60=C[i - 1] / C[i - 61] - 1,
                 vol60=np.nanstd(lr[i - 60:i]), dd250=C[i - 1] / h250, hi250_break=float(C[i] > h250),
                 spike60=int(np.nansum(np.abs(lr[i - 60:i]) >= np.log(1.15))), up20=float(np.nanmean(lr[i - 20:i] > 0)),
                 hist_days=int(valid[:i].sum()))
        lb = label(O, H, L, C, V, valid, cal, i, ss, LAST)
        if src == "old":
            r.update(labeled=lab_old, y3x=(y_old == True) if lab_old else np.nan, lab_check=lb.get("y3x"), lab_valid_mine=lb["lab_valid"])
        else:
            r.update(labeled=lb["lab_valid"], y3x=lb.get("y3x", np.nan) if lb["lab_valid"] else np.nan, lab_check=lb.get("y3x"), lab_valid_mine=lb["lab_valid"])
        r.update(total_mult=lb.get("total_mult", np.nan), follow_mult=lb.get("follow_mult", np.nan))
        for day, k in (("d1", 1), ("d2", 2)):
            e = i + k
            if e >= N or not np.isfinite(O[e]) or O[e] <= 0: continue
            for Hh in (20, 40, 60):
                jj = e + Hh - 1; r[f"{day}_tcomp{Hh}"] = jj <= N - 1; jj = min(jj, N - 1)
                w = C[e:jj + 1]
                r[f"{day}_tup{Hh}"] = float(C[jj] > O[e]); r[f"{day}_tret{Hh}"] = C[jj] / O[e] - 1
                r[f"{day}_cmax{Hh}"] = np.nanmax(w) / O[e] - 1; r[f"{day}_cmin{Hh}"] = np.nanmin(w) / O[e] - 1
        rows.append(r)
E = pd.DataFrame(rows).sort_values(["t0", "symbol"]).reset_index(drop=True)
E["year"] = E.t0.dt.year
E["per"] = np.where(E.t0 < "2024-01-01", "disc", np.where(E.t0 < "2026-01-01", "valid", "test"))
E.to_pickle(OUT)
o = E[E.src == "old"]
print("events", len(E), "| new", int((E.src == "new").sum()), "old", len(o), "(stored 5083)")
print("old label check: valid agree", int((o.labeled == o.lab_valid_mine).sum()), "/", len(o), "| y3x agree on labeled", int(((o.y3x == True) == (o.lab_check == True))[o.labeled == True].sum()), "/", int((o.labeled == True).sum()))
print(E.groupby("year").agg(n=("symbol", "size"), labeled=("labeled", "sum"), ge=("y3x", lambda x: int((x == True).sum()))).T.to_string())
