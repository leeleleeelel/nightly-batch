# Daily scoring with the FROZEN attempt-25 model (a25_model.pkl): stage-1 rules with the stored 2019-2023 thresholds, stage-2 logistic score,
# live pick rule = score >= 90th percentile of the scores of all candidates in the 365 calendar days before (attempt 39). No refitting.
# usage: score_daily.py CORE MODEL_PKL OUT_PKL
import sys, os, re, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a25_train2 import assemble
CORE, MP, OUT = sys.argv[1:4]
M = pickle.load(open(MP, "rb"))
E, X0, FT, REL = assemble(CORE, E_filter=False)
X0 = X0.reindex(columns=M["keep0"]).astype(np.float32); XC = pd.concat([X0, REL], axis=1)
def cmask(c):
    f, side = re.match(r"^(.*) (top10|bot10|=1|=0)$", c.strip()).groups()
    if f not in XC.columns: return np.zeros(len(E), bool)
    v = XC[f].to_numpy(float)
    if side in ("=1", "=0"): return v == float(side[1])
    if f not in M["stage1_thresholds"]: return np.zeros(len(E), bool)
    q10, q90 = M["stage1_thresholds"][f]; m = v >= q90 if side == "top10" else v <= q10; return np.where(np.isfinite(v), m, False)
S = np.zeros(len(E), bool); hit = [[] for _ in range(len(E))]
for k, rule in enumerate(M["rules"]):
    m = np.ones(len(E), bool)
    for c in rule.split(" & "): m &= cmask(c)
    S |= m
    for q in np.where(m)[0]: hit[q].append(k)
XA = pd.concat([X0.drop(columns=M["leak"], errors="ignore"), FT, REL], axis=1).reindex(columns=M["cols"])
sc = np.full(len(E), np.nan); sc[S] = M["model"].predict_proba(XA[S])[:, 1]
t0 = E.t0.to_numpy(); ci = np.where(S)[0]; th = np.full(len(E), np.nan)
for k in ci:
    w = ci[(t0[ci] < t0[k].astype("datetime64[D]")) & (t0[ci] >= t0[k] - np.timedelta64(365, "D"))]
    if len(w) >= 50: th[k] = np.nanpercentile(sc[w], 90)
pick = S & np.isfinite(th) & (sc >= th)
# per-pick explanation: the model's largest standardized contributions (coefficient x standardized value)
imp, scl, lr = M["model"].named_steps["simpleimputer"], M["model"].named_steps["standardscaler"], M["model"].named_steps["logisticregression"]
Z = np.full((len(E), len(M["cols"])), np.nan, np.float32)
if S.any(): Z[S] = scl.transform(imp.transform(XA[S])).astype(np.float32)
contrib = Z * lr.coef_[0][None, :].astype(np.float32)
top_up, top_dn = [None] * len(E), [None] * len(E)
for q in np.where(pick)[0]:
    o = np.argsort(contrib[q]); top_up[q] = [(M["cols"][i], float(contrib[q][i]), float(XA.iloc[q, i])) for i in o[::-1][:6]]; top_dn[q] = [(M["cols"][i], float(contrib[q][i]), float(XA.iloc[q, i])) for i in o[:4]]
R = pd.DataFrame(dict(symbol=E.symbol, t0=E.t0, i=E.i, j=E.j, cand=S, score=sc, threshold=th, pick=pick, rules_hit=hit, top_up=top_up, top_dn=top_dn,
                      labeled=E.labeled, comp20=E.get("e1_comp_20"), y10=(E.e1_hold2_20 >= 0.10).where(E.e1_comp_20 == True)), index=E.index)
R.to_pickle(OUT); last = R.t0.max()
print("last session", last.date(), "| surges", int((R.t0 == last).sum()), "candidates", int((R.cand & (R.t0 == last)).sum()), "picks", int((R.pick & (R.t0 == last)).sum()))
