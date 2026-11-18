# Attempt 25 rebuilt on the rebuilt core: same feature assembly, frozen stage-1 rule list (attempt 22 golden-egg rules), stage-2 logistic C=0.01.
# Saves the artifacts the daily pipeline needs (feature columns, stage-1 thresholds from 2019-2023, fitted model) and the scores, and compares
# aggregate numbers with the original run (1,033/736/300 candidates; 74/30 holdout picks at 63.5%/53.3%; OOF top-10% 57%).
# usage: a25_train2.py CORE OUT_DIR
import sys, re, json, pickle, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); CORE, OUT = sys.argv[1], sys.argv[2]
def assemble(CORE, E_filter=True):
    E = pd.read_pickle(CORE + "/events_x19.pkl").join(pd.read_pickle(CORE + "/feat_a.pkl")).join(pd.read_pickle(CORE + "/feat_b.pkl")).join(pd.read_pickle(CORE + "/capture.pkl"))
    MS = pd.read_pickle(CORE + "/market_state.pkl").drop(columns=["regime"], errors="ignore"); MS.columns = [f"mkt_{c}" for c in MS.columns]
    E = E.join(MS.reindex(E.t0).set_index(E.index)); Q = pd.read_pickle(CORE + "/wq_T0.pkl")
    if E_filter: E = E[(E.labeled == True) & (E.e1_comp_20 == True)].copy()
    DROP = {"i", "j", "year", "labeled", "y3x", "lab_check", "lab_valid_mine", "total_mult", "follow_mult", "t0_open", "t0_close", "price_pre", "hist_days", "offer_after5", "cik"}
    num = [c for c in E.columns if c not in DROP and not c.startswith(("d1_", "d2_", "e1_", "e3_")) and pd.api.types.is_numeric_dtype(E[c])]
    X0 = pd.concat([E[num].astype(float), Q.reindex(E.index).astype(float)], axis=1).replace([np.inf, -np.inf], np.nan)
    FT = pd.read_pickle(CORE + "/ftd_feat.pkl").reindex(E.index).astype(np.float32); REL = pd.read_pickle(CORE + "/rel_feat.pkl").reindex(E.index).astype(np.float32)
    return E, X0, FT, REL
if __name__ == "__main__":
    E, X0, FT, REL = assemble(CORE)
    keep0 = X0.columns[X0.notna().mean() > 0.05]; X0 = X0[keep0].astype(np.float32)
    yr = E.t0.dt.year.to_numpy(); disc = yr <= 2023; per = np.where(disc, "2019-23", np.where(yr <= 2025, "2024-25", "2026"))
    XC = pd.concat([X0, REL], axis=1); RG = pd.read_csv(CORE + "/rel_rules_ge.csv"); TH = {}
    def cmask(c):
        f, side = re.match(r"^(.*) (top10|bot10|=1|=0)$", c.strip()).groups()
        if f not in XC.columns: return np.zeros(len(E), bool)
        v = XC[f].to_numpy(float)
        if side in ("=1", "=0"): return v == float(side[1])
        if f not in TH: TH[f] = tuple(np.nanpercentile(v[disc], [10, 90]))
        q10, q90 = TH[f]; m = v >= q90 if side == "top10" else v <= q10; return np.where(np.isfinite(v), m, False)
    S = np.zeros(len(E), bool); miss = set()
    for rule in RG["rule"]:
        m = np.ones(len(E), bool)
        for c in rule.split(" & "):
            f = re.match(r"^(.*) (top10|bot10|=1|=0)$", c.strip()).group(1)
            if f not in XC.columns: miss.add(f)
            m &= cmask(c)
        S |= m
    LEAK = [c for c in ("own_ge180", "own_maxmult180") if c in X0.columns]
    XA = pd.concat([X0.drop(columns=LEAK), FT, REL], axis=1); cols = list(XA.columns)
    y = (E.e1_hold2_20.to_numpy() >= 0.10).astype(int)
    def fit(Xtr, ytr): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced")).fit(Xtr, ytr)
    oof = np.full(len(E), np.nan)
    for Yy in (2021, 2022, 2023):
        tr = S & disc & (yr < Yy); te = S & (yr == Yy); m = fit(XA[tr], y[tr]); oof[te] = m.predict_proba(XA[te])[:, 1]
    fin = fit(XA[S & disc], y[S & disc]); sc = np.full(len(E), np.nan); sc[S] = fin.predict_proba(XA[S])[:, 1]
    out = dict(candidates={b: int((S & (per == b)).sum()) for b in ("2019-23", "2024-25", "2026")}, missing_rule_features=sorted(miss), n_features=len(cols), n_rules=len(RG))
    for b in ("2024-25", "2026"):
        mm = S & (per == b); th = np.nanpercentile(sc[mm], 90); p = mm & (sc >= th); out[f"hold_{b}"] = dict(n=int(p.sum()), y10=float(y[p].mean()))
    pk = np.zeros(len(E), bool)
    for Yy in (2021, 2022, 2023):
        mm = S & (yr == Yy); th = np.nanpercentile(oof[mm], 90); pk |= mm & (oof >= th)
    out["oof_top10"] = dict(n=int(pk.sum()), y10=float(y[pk].mean()))
    out["original"] = dict(candidates={"2019-23": 1033, "2024-25": 736, "2026": 300}, hold_2024_25=dict(n=74, y10=0.635), hold_2026=dict(n=30, y10=0.533), oof_top10=dict(n=68, y10=0.574))
    pickle.dump(dict(model=fin, cols=cols, keep0=list(keep0), leak=LEAK, stage1_thresholds=TH, rules=list(RG["rule"])), open(OUT + "/a25_model.pkl", "wb"))
    pd.DataFrame(dict(symbol=E.symbol, t0=E.t0, per=per, cand=S, oof=oof, score=sc, y10=y), index=E.index).to_pickle(OUT + "/a25_scores_rebuilt.pkl")
    json.dump(out, open(OUT + "/a25_rebuild_check.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
