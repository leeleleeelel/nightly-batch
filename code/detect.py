# Re-implementation of the registered t0 trigger (event_study.py) on a common session calendar.
import numpy as np, pandas as pd
def prep(df, cal):
    f = df.set_index("date")[["o", "h", "l", "c", "v"]]
    f = f[~f.index.duplicated(keep="last")].reindex(cal)
    a = f.to_numpy(dtype=float); O, H, L, C, V = (a[:, i] for i in range(5))
    valid = np.isfinite(a).all(axis=1) & (O > 0) & (H > 0) & (L > 0) & (C > 0) & (V >= 0) & \
            (H >= np.maximum(O, C) * (1 - 1e-5)) & (L <= np.minimum(O, C) * (1 + 1e-5)) & (H >= L)
    return O, H, L, C, V, valid
def triggers(O, H, L, C, V, valid, cal, start, end, cooldown=22):
    D = C * V
    pv = pd.Series(V).rolling(20, min_periods=20).median().shift(1).to_numpy()
    pdm = pd.Series(D).rolling(20, min_periods=20).median().shift(1).to_numpy()
    pc = pd.Series(C).rolling(20, min_periods=20).max().shift(1).to_numpy()
    hv = pd.Series(valid.astype(int)).rolling(61, min_periods=61).sum().to_numpy() == 61
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = C / np.r_[np.nan, C[:-1]]; r3 = C / np.r_[np.nan, np.nan, np.nan, C[:-3]]
    cand = hv & (cal >= start) & (cal <= end) & (pv > 0) & (pdm >= 250000) & (D >= 2000000) & ((r1 >= 1.15) | (r3 >= 1.30)) & (C > pc) & (V >= 2 * pv)
    out, last = [], -10000
    for i in np.flatnonzero(cand):
        if not valid[i - 60:i + 1].all(): continue
        if i - last <= cooldown: continue
        last = i; out.append(i)
    return out
