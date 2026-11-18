# attempt-42 rule matching for recent events (no search, no outcome use): evaluates the kept reach_20 rules' formulas with point-in-time prices.
# usage: a42_match.py CORE RULES_JSON SCORES_PKL OUT_JSON
import sys, json, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
CORE, RJ, SCP, OUT = sys.argv[1:5]
RU = json.load(open(RJ))["rules"]; S0 = pd.read_pickle(SCP)
E = pd.read_pickle(CORE + "/events_x19.pkl"); last = E.t0.max()
sel = E.index[(E.t0 >= last - pd.Timedelta(days=60)) | E.index.isin(S0.index[S0.pick & (S0.t0 >= "2026-11-05")])]
E = E.loc[sel]; n = len(E); I = E.i.to_numpy(); J = E.j.to_numpy(); L = 120; IDX = np.maximum(I[:, None] + np.arange(-L + 1, 1)[None, :], 0)
syms = json.load(open(CORE + "/px/syms.json")); cal = pd.DatetimeIndex(pd.to_datetime(np.load(CORE + "/px/cal.npy", allow_pickle=True)))
Om = np.load(CORE + "/px/o.npy"); Cm = np.load(CORE + "/px/c.npy")
SP = pd.read_csv(CORE + "/px/splits_x19.csv"); SP["date"] = pd.to_datetime(SP.date); SP["f"] = SP.numerator / SP.denominator
by = {s: (g.date.values, g.f.values) for s, g in SP.groupby("symbol")}; t0d = E.t0.to_numpy(); Fpit = np.ones(n)
for q in range(n):
    g = by.get(syms[J[q]])
    if g is None: continue
    a = g[0] > t0d[q]
    if a.any(): Fpit[q] = float(np.prod(g[1][a]))
Cp = np.vstack([np.full((1, Cm.shape[1]), np.nan, np.float32), Cm[:-1]]); mk = np.nanmean(np.clip(Cm / Cp - 1, -0.5, 1.0), axis=1).astype(np.float32); del Cp
SER = {"C": Cm[IDX, J[:, None]].astype(np.float32), "O": Om[IDX, J[:, None]].astype(np.float32)}; del Om, Cm
for k_, f_ in (("H", "h"), ("L", "l"), ("V", "v")):
    A = np.load(CORE + f"/px/{f_}.npy", mmap_mode="r"); SER[k_] = np.asarray(A[IDX, J[:, None]], np.float32); del A
for k_ in "OHLC": SER[k_] = (SER[k_] * Fpit[:, None]).astype(np.float32)
SER["V"] = (SER["V"] / Fpit[:, None]).astype(np.float32)
A = np.load(CORE + "/svr.npy", mmap_mode="r"); SER["S"] = np.asarray(A[IDX, J[:, None]], np.float32); del A
SER["M"] = mk[IDX].astype(np.float32)
# ---------------- operators (unchanged) ----------------
def shift(x, k):
    out = np.full_like(x, np.nan); out[:, k:] = x[:, :-k]; return out
def rsum(x, wd):
    m = np.isfinite(x); x0 = np.where(m, x, 0.0).astype(np.float64); c = np.cumsum(x0, 1); cn = np.cumsum(m, 1)
    s = np.full(x.shape, np.nan); s[:, wd - 1] = c[:, wd - 1]; s[:, wd:] = c[:, wd:] - c[:, :-wd]
    nn = np.full(x.shape, 0.0); nn[:, wd - 1] = cn[:, wd - 1]; nn[:, wd:] = cn[:, wd:] - cn[:, :-wd]
    return s, nn
def rmean(x, wd):
    s, nn = rsum(x, wd); return np.where(nn >= 0.7 * wd, s / np.maximum(nn, 1), np.nan)
def rstd(x, wd):
    m1 = rmean(x, wd); m2 = rmean(x * x, wd); v = m2 - m1 * m1; return np.sqrt(np.where(v > 0, v, np.nan))
def rcorr(x, y, wd):
    mx, my = rmean(x, wd), rmean(y, wd); cxy = rmean(x * y, wd) - mx * my; sx, sy = rstd(x, wd), rstd(y, wd); return np.clip(cxy / (sx * sy), -1, 1)
BASES = ["O", "H", "L", "C", "V", "S", "M"]; WS = [3, 5, 10, 20, 40, 60]; KS = [1, 2, 3, 5, 10, 20]
G = np.random.default_rng(20261111)
def gen_unary():
    b = str(G.choice(BASES)); t = str(G.choice(["id", "log", "ret", "diff"] if b in "OHLCV" else ["id", "diff"])); k = int(G.choice(KS))
    op = str(G.choice(["mean", "std", "z", "sum", "delay", "abs", "sign"])) if G.random() < 0.6 else None; p = int(G.choice(WS)) if op in ("mean", "std", "z", "sum") else int(G.choice(KS))
    return (b, t, k, op, p)
def gen_formula():
    a = gen_unary(); node = ("bin", str(G.choice(["add", "sub", "mul", "div", "corr", "max", "min"])), a, gen_unary(), int(G.choice(WS[1:]))) if G.random() < 0.5 else a
    return (str(G.choice(["last", "rank", "argmax", "argmin", "max", "min", "mean", "delta"])), node, int(G.choice(WS)), int(G.choice(KS)))
def ustr(u):
    b, t, k, op, p = u; s = {"id": b, "log": f"log({b})", "ret": f"chg{k}({b})", "diff": f"diff{k}({b})"}[t]
    return s if op is None else (f"{op}{p}({s})" if op in ("mean", "std", "z", "sum", "delay") else f"{op}({s})")
def fstr(f):
    red, node, rw, rk = f; inner = ustr(node) if node[0] != "bin" else (f"corr{node[4]}({ustr(node[2])}, {ustr(node[3])})" if node[1] == "corr" else f"{node[1]}({ustr(node[2])}, {ustr(node[3])})")
    return {"last": f"t0[{inner}]", "rank": f"rank{rw}[{inner}]", "argmax": f"days_since_max{rw}[{inner}]", "argmin": f"days_since_min{rw}[{inner}]",
            "max": f"max{rw}[{inner}]", "min": f"min{rw}[{inner}]", "mean": f"mean{rw}[{inner}]", "delta": f"delta{rk}[{inner}]"}[red]
def ueval(u):
    b, t, k, op, p = u; x = SER[b].astype(np.float64)
    if t == "log": x = np.where(x > 0, np.log(np.where(x > 0, x, 1)), np.nan)
    elif t == "ret": x = x / shift(x, k) - 1
    elif t == "diff": x = x - shift(x, k)
    if op == "mean": x = rmean(x, p)
    elif op == "std": x = rstd(x, p)
    elif op == "z": x = (x - rmean(x, p)) / rstd(x, p)
    elif op == "sum": x = rsum(x, p)[0]
    elif op == "delay": x = shift(x, p)
    elif op == "abs": x = np.abs(x)
    elif op == "sign": x = np.sign(x)
    return x
def feval(f):
    red, node, rw, rk = f
    if node[0] == "bin":
        a, b = ueval(node[2]), ueval(node[3]); c = node[1]
        v = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "div": lambda: np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1), np.nan),
             "corr": lambda: rcorr(a, b, node[4]), "max": lambda: np.fmax(a, b), "min": lambda: np.fmin(a, b)}[c]()
    else:
        v = ueval(node)
    v = np.where(np.isfinite(v), v, np.nan); seg = v[:, -rw:]; last = v[:, -1]
    if red == "last": out = last
    elif red == "rank": out = np.where(np.isfinite(last), (seg <= last[:, None]).sum(1) / np.maximum(np.isfinite(seg).sum(1), 1), np.nan)
    elif red == "argmax": out = np.where(np.isfinite(seg).any(1), rw - 1 - np.argmax(np.where(np.isfinite(seg), seg, -np.inf), 1), np.nan)
    elif red == "argmin": out = np.where(np.isfinite(seg).any(1), rw - 1 - np.argmin(np.where(np.isfinite(seg), seg, np.inf), 1), np.nan)
    elif red == "max": out = np.nanmax(seg, 1)
    elif red == "min": out = np.nanmin(seg, 1)
    elif red == "mean": out = np.nanmean(seg, 1)
    else: out = last - v[:, -1 - rk]
    return out.astype(np.float32)
BN = {"O": "시가", "H": "고가", "L": "저가", "C": "종가", "V": "거래량", "S": "공매도 거래 비율", "M": "시장 평균 수익률"}
def split_args(s):
    depth = 0
    for k, ch in enumerate(s):
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif ch == "," and depth == 0: return s[:k].strip(), s[k + 1:].strip()
    return s, None
def kr_u(s):
    s = s.strip()
    if s in BN: return BN[s]
    m = re.match(r"^(log|chg|diff|mean|std|z|sum|delay|abs|sign)(\d*)\((.*)\)$", s)
    if not m: return s
    op, k, inner = m.groups(); x = kr_u(inner)
    return {"log": f"{x}(로그)", "chg": f"{x}의 {k}일 변화율", "diff": f"{x}의 {k}일 차이", "mean": f"{x}의 {k}일 평균", "std": f"{x}의 {k}일 변동성", "z": f"{x}의 {k}일 기준 z점수",
            "sum": f"{x}의 {k}일 합", "delay": f"{k}일 전 {x}", "abs": f"{x}의 크기", "sign": f"{x}의 부호"}[op]
def kr_node(s):
    m = re.match(r"^(corr\d+|add|sub|mul|div|max|min)\((.*)\)$", s)
    if m:
        a, b = split_args(m.group(2))
        if b is not None:
            op = m.group(1); A, B = kr_u(a), kr_u(b)
            if op.startswith("corr"): return f"[{A}]·[{B}]의 {op[4:]}일 상관"
            return f"[{A}] {dict(add='+', sub='−', mul='×', div='÷')[op]} [{B}]" if op in ("add", "sub", "mul", "div") else f"[{A}]·[{B}] 중 {'큰' if op == 'max' else '작은'} 값"
    return kr_u(s)
def kr(f):
    m = re.match(r"^(t0|rank|days_since_max|days_since_min|max|min|mean|delta)(\d*)\[(.*)\]$", f)
    if not m: return f
    red, w, inner = m.groups(); x = kr_node(inner)
    return {"t0": f"급등일의 {x}", "rank": f"최근 {w}일 중 급등일 {x}의 순위", "days_since_max": f"최근 {w}일 중 「{x}」 값이 가장 컸던 날이 며칠 전인가", "days_since_min": f"최근 {w}일 중 「{x}」 값이 가장 작았던 날이 며칠 전인가",
            "max": f"최근 {w}일 {x}의 최댓값", "min": f"최근 {w}일 {x}의 최솟값", "mean": f"최근 {w}일 {x}의 평균", "delta": f"「{x}」의 급등일 값과 {w}일 전 값의 차이"}[red]
GEN = {}
for _ in range(20000):
    f = gen_formula(); GEN.setdefault(fstr(f), f)
cache = {}; out = {}
def val(fs):
    if fs not in cache: cache[fs] = feval(GEN[fs])
    return cache[fs]
for r in RU:
    m = np.ones(n, bool)
    for c in r["conds"]:
        v = val(c["formula"]); m &= np.where(np.isfinite(v), v >= c["q"] if c["side"] == "top10" else v <= c["q"], False)
    desc = " 그리고 ".join(f"{kr(c['formula'])} {'상위 10%' if c['side'] == 'top10' else '하위 10%'}" for c in r["conds"])
    rates = f"과거 +20% 도달: 2019~21 {100 * r['r20_D1']:.0f}% ({r['n_D1']}건) · 2022~23 {100 * r['r20_D2']:.0f}% · 2024~25 {100 * r['r20_2024-25']:.0f}% · 2026 {100 * r['r20_2026']:.0f}%" if r.get("n_D1") else ""
    for q in np.where(m)[0]: out.setdefault(str(E.index[q]), []).append(desc + " — " + rates)
json.dump(dict(matches=out, n_rules=len(RU), evaluated=n), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("rules", len(RU), "events", n, "with matches", len(out))
