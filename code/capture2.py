# REBUILD 2026-11-18: unchanged copy, run on data/core.
# "Could you have sold at a profit at a realistic moment?" for every surge event: after the entry open, the best level the stock held on TWO
# consecutive daily closes (an intraday or one-close spike does not count; seeing the first close and selling at the next close still locks it in).
# Entries: t0+1 open (decision at the t0 close) and t0+3 open (decision at the t0+2 close). Windows 5/10/20/60 sessions. No stop, no holding rule.
import sys, json, numpy as np, pandas as pd
CORE = sys.argv[1]
E = pd.read_pickle(CORE + "/events_x19.pkl"); N = len(np.load(CORE + "/px/cal.npy"))
O = np.ascontiguousarray(np.load(CORE + "/px/o.npy").T).astype(np.float64); C = np.ascontiguousarray(np.load(CORE + "/px/c.npy").T).astype(np.float64)
J, I = E.j.to_numpy(), E.i.to_numpy(); out = {}
for tag, k in (("e1", 1), ("e3", 3)):
    e = I + k; ent = np.where(e < N, O[J, np.minimum(e, N - 1)], np.nan)
    for H in (5, 10, 20, 60):
        idx = e[:, None] + np.arange(H)[None, :]; ok = idx < N; w = C[J[:, None], np.minimum(idx, N - 1)]; w[~ok] = np.nan
        two = np.fmin(w[:, :-1], w[:, 1:])                                   # level held on two consecutive closes
        out[f"{tag}_hold2_{H}"] = np.nanmax(two, axis=1) / ent - 1           # best two-close level vs entry
        out[f"{tag}_maxc_{H}"] = np.nanmax(w, axis=1) / ent - 1              # best single close (reference)
        out[f"{tag}_comp_{H}"] = (e + H - 1) <= N - 1
X = pd.DataFrame(out, index=E.index); X.to_pickle(CORE + "/capture.pkl")
L = E.join(X)[(E.labeled == True)]
for blk, m in (("2019-23", L.year <= 2023), ("2024-25", L.year.between(2024, 2025)), ("2026", L.year == 2026)):
    d = L[m & L.e1_comp_20]
    print(blk, "all surges, entry t0+1 open, within 20 sessions, two closes >= +5/10/20/50/100%:", [round(float((d.e1_hold2_20 >= g).mean()), 3) for g in (0.05, 0.10, 0.20, 0.50, 1.0)])
