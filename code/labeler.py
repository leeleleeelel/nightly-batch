# Golden-egg label (v2 total_triple_followthrough) re-implemented from labels_build.py, plus the cohort quality checks
# (t-60..t+25 unexplained gaps, splits t0-30d..t+25, valid bars, entry-day volume). Pure price arithmetic.
import numpy as np, pandas as pd
def label(O, H, L, C, V, valid, cal, i, split_dates, last_obs):
    N = len(cal)
    if i + 25 >= N: return dict(lab_valid=False, lab_reason="short_window")
    if i < 61: return dict(lab_valid=False, lab_reason="short_history")
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = O[i - 60:i + 26] / C[i - 61:i + 25]
    if np.any(np.isfinite(gap) & ((gap >= 4) | (gap <= 0.25))): return dict(lab_valid=False, lab_reason="gap")
    d0 = cal[i]; d25 = cal[i + 25]
    if any(d0 - pd.Timedelta(days=30) <= s <= d25 for s in split_dates): return dict(lab_valid=False, lab_reason="split")
    if not valid[i - 60:i + 26].all(): return dict(lab_valid=False, lab_reason="invalid_bar")
    a = i - 3; deadline = cal[a] + pd.Timedelta(days=31)
    if deadline > last_obs: return dict(lab_valid=False, lab_reason="censored")
    k = int(np.searchsorted(cal.values, np.datetime64(deadline), side="right") - 1)
    if k <= i or not valid[a:k + 1].all(): return dict(lab_valid=False, lab_reason="invalid_window")
    if V[i + 1] <= 0: return dict(lab_valid=False, lab_reason="zero_volume_entry")
    w = C[i + 1:k + 1]; up = C[i + 1:k + 1] > C[i:k]
    peak = float(np.max(w)); total = peak >= 3.0 * C[a]; cont = peak >= 1.25 * C[i]
    return dict(lab_valid=True, lab_reason="", y3x=bool(total and cont and int(up.sum()) >= 2), total_mult=peak / C[a], follow_mult=peak / C[i],
                up_closes=int(up.sum()), end_k=k)
