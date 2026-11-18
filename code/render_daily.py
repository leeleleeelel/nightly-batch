# Daily signal page (Korean, mobile-friendly): today's picks of the frozen attempt-25 live rule with expected probabilities and target prices
# (prereg_target.json logic A, plus attempt-42 matched combinations when that search passed), reasons, the rule's historical stats under
# realistic exits, today's other candidates, and the running record of every signal since 2026-11-05.
# usage: render_daily.py CORE SCORES_PKL OUT_DIR [A42_MATCHES_JSON]
import sys, os, json, html, datetime, numpy as np, pandas as pd
CORE, SCP, OUTD = sys.argv[1:4]; A42 = sys.argv[4] if len(sys.argv) > 4 and os.path.exists(sys.argv[4]) else None
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    from rel_names import fname
except Exception:
    fname = lambda f: f
esc = html.escape
pp = lambda x: "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.0f}%"
s1 = lambda x: "–" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%".replace("-", "−")
R = pd.read_pickle(SCP); cal = pd.DatetimeIndex(np.load(CORE + "/px/cal.npy")); N = len(cal)
P = {k: np.load(CORE + f"/px/{k}.npy", mmap_mode="r") for k in "ohlc"}
LEV = (0.10, 0.15, 0.20, 0.30, 0.50)
def outcome(i, j):
    rows = np.arange(i + 1, min(i + 21, N))
    if len(rows) == 0: return dict(started=False)
    o, h, l, c = (np.asarray(P[k][rows, j], float) for k in "ohlc"); e = o[0]; n = len(c)
    r = c / e - 1; hi = h / e - 1; lo = l / e - 1; op = o / e - 1
    two = np.where(np.fmin(r[:-1], r[1:]) >= 0.10)[0] if n >= 2 else np.array([], int)
    exA = float(r[two[0] + 1]) if len(two) else float(r[-1])
    t = np.where(hi >= 0.10)[0]; exB = (max(op[t[0]], 0.10) if t[0] > 0 else 0.10) if len(t) else float(r[-1])
    exC = float(r[-1])                                                     # +10% limit with a -20% stop (same day: the level nearer the open first)
    for d in range(n):
        if d > 0 and op[d] >= 0.10: exC = op[d]; break
        if d > 0 and op[d] <= -0.20: exC = op[d]; break
        up, dn = hi[d] >= 0.10, lo[d] <= -0.20
        if up and dn: exC = 0.10 if (0.10 - op[d]) <= (op[d] + 0.20) else -0.20; break
        if up: exC = 0.10; break
        if dn: exC = -0.20; break
    return dict(started=True, n=n, complete=n >= 20, entry=e, y10=bool(len(two)), reach={L: bool((hi >= L).any()) for L in LEV},
                ret=float(r[-1]), mae=float(lo.min()), mfe=float(hi.max()), exA=exA, exB=float(exB), exC=float(exC))
# ---- history: out-of-sample picks with complete outcomes (OOF top-10% 2021-23 + frozen-model live picks from 2024) ----
SRP = os.path.join(os.path.dirname(HERE), "a25_scores_rebuilt.pkl"); SR = pd.read_pickle(SRP) if os.path.exists(SRP) else None
hist = []
if SR is not None:
    yr = SR.t0.dt.year.to_numpy(); oof = SR.oof.to_numpy()
    for Y in (2021, 2022, 2023):
        mm = SR.cand.to_numpy() & (yr == Y) & np.isfinite(oof)
        if mm.any(): hist += [(q, "교차검증") for q in SR.index[mm & (oof >= np.nanpercentile(oof[mm], 90))]]
seen = {q for q, _ in hist}; hist += [(q, "고정 모델") for q in R.index[R.pick & (R.t0 >= "2024-01-01")] if q not in seen]
H = []
for q, src in hist:
    if q not in R.index: continue
    o = outcome(int(R.at[q, "i"]), int(R.at[q, "j"]))
    if o.get("complete"): H.append(dict(idx=q, t0=R.at[q, "t0"], src=src, y10=o["y10"], exA=o["exA"], exB=o["exB"], exC=o["exC"], **{f"r{int(L * 100)}": o["reach"][L] for L in LEV}))
H = pd.DataFrame(H)
def rate(col, d=None):
    d = H if d is None else d; n = len(d)
    if n == 0: return dict(p=np.nan, n=0, lo=np.nan, hi=np.nan)
    p = float(d[col].mean()); z = 1.645; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den; half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(p=p, n=n, lo=ctr - half, hi=ctr + half)
recent = H[H.t0 >= H.t0.max() - pd.Timedelta(days=365)] if len(H) else H
PR = {k: rate(k) for k in ["y10"] + [f"r{int(L * 100)}" for L in LEV]}; PRr = {k: rate(k, recent) for k in PR}
t1 = max([L for L in LEV if PR[f"r{int(L * 100)}"]["p"] >= 0.70], default=None); t2 = max([L for L in LEV if PR[f"r{int(L * 100)}"]["p"] >= 0.50], default=None)
prob_rows = (f"<tr><td>종가 기준 성공 (20거래일 안 종가 이틀 연속 +10%)</td><td><b>{pp(PR['y10']['p'])}</b></td><td class=dim>{pp(PR['y10']['lo'])}~{pp(PR['y10']['hi'])}</td><td>{pp(PRr['y10']['p'])}</td></tr>"
             + "".join(f"<tr><td>장중 +{int(L * 100)}%에 닿음</td><td><b>{pp(PR[f'r{int(L * 100)}']['p'])}</b></td><td class=dim>{pp(PR[f'r{int(L * 100)}']['lo'])}~{pp(PR[f'r{int(L * 100)}']['hi'])}</td><td>{pp(PRr[f'r{int(L * 100)}']['p'])}</td></tr>" for L in LEV))
def ex_row(lab, col):
    if not len(H): return ""
    x = H[col].to_numpy(); return f"<tr><td>{lab}</td><td><b>{pp((x > 0).mean())}</b></td><td><b>{s1(x.mean())}</b></td><td>{s1(np.median(x))}</td><td>{s1(x.min())}</td></tr>"
ex_rows = ex_row("A. 종가가 이틀 연속 +10%면 그 종가에 매도, 아니면 20일째 종가", "exA") + ex_row("B. +10%에 지정가 매도, 안 닿으면 20일째 종가", "exB") + ex_row("C. B + −20% 손절", "exC")
by_year = "".join(f"<tr><td>{Y}</td><td>{int((H.t0.dt.year == Y).sum())}</td><td>{pp(H[H.t0.dt.year == Y].y10.mean())}</td><td>{pp(H[H.t0.dt.year == Y].r10.mean())}</td><td>{s1(H[H.t0.dt.year == Y].exB.mean())}</td></tr>" for Y in sorted(H.t0.dt.year.unique())) if len(H) else ""
A42M = json.load(open(A42, encoding="utf-8")).get("matches", {}) if A42 else None
# ---- today ----
last = R.t0.max(); today = R[R.t0 == last]; picks = today[today.pick]; others = today[today.cand & ~today.pick]
tg = lambda L, c0: f"+{int(L * 100)}% <span class=dim>(약 ${c0 * (1 + L):,.2f})</span>" if L else "–"
def pick_card(q, r):
    c0 = float(P["c"][int(r.i), int(r.j)]); up = r.top_up or []; dn = r.top_dn or []
    why = "".join(f"<li>{esc(fname(f))}</li>" for f, cval, x in up[:5]); against = "".join(f"<li>{esc(fname(f))}</li>" for f, cval, x in dn[:3])
    mt = ""
    if A42M is not None:
        hits = A42M.get(str(q), []); mt = "<p><b>해당하는 +20% 조합(시도 42):</b></p>" + ("<ul>" + "".join(f"<li>{esc(h)}</li>" for h in hits[:5]) + "</ul>" if hits else "<p class=dim>없음</p>")
    return f"""<div class=card><h3>{esc(r.symbol)} <span class=dim>급등일 {r.t0:%m-%d} · 직전 종가 ${c0:,.2f}</span></h3>
<p><b>예상 확률:</b> 종가 기준 성공 {pp(PR['y10']['p'])} · 장중 +10% {pp(PR['r10']['p'])} · +20% {pp(PR['r20']['p'])} · +30% {pp(PR['r30']['p'])}</p>
<p><b>목표가:</b> 1차 {tg(t1, c0)} · 2차 {tg(t2, c0)} <span class=dim>(직전 종가 기준 예상; 실제 기준은 다음 날 시가)</span></p>
<p class=dim>점수 {r.score:.3f} (문턱 {r.threshold:.3f}) · 황금알 흔적 규칙 {len(r.rules_hit)}개에 해당</p>
<p><b>점수를 올린 근거</b></p><ul>{why}</ul><p><b>점수를 깎은 쪽</b></p><ul>{against}</ul>{mt}</div>"""
cards = "".join(pick_card(q, r) for q, r in picks.iterrows()) or "<div class=card><p>오늘은 규칙이 고른 종목이 없습니다.</p></div>"
oth = "".join(f"<tr><td>{esc(r.symbol)}</td><td>{r.score:.3f}</td><td>{r.threshold:.3f}</td></tr>" for q, r in others.sort_values("score", ascending=False).iterrows())
oth_tbl = f"<div class=scrollx><table><tr><th>후보(고르지 않음)</th><th>점수</th><th>문턱</th></tr>{oth}</table></div>" if oth else "<p class=dim>없음</p>"
rec = R[R.pick & (R.t0 >= "2026-11-05")].sort_values("t0", ascending=False); rrows = ""
for q, r in rec.iterrows():
    o = outcome(int(r.i), int(r.j))
    if not o.get("started"): rrows += f"<tr><td>{r.t0:%m-%d}</td><td>{esc(r.symbol)}</td><td colspan=5>진입 전</td></tr>"; continue
    rr = " ".join(f"+{int(L * 100)}%{'✓' if o['reach'][L] else '·'}" for L in LEV[:4])
    rrows += f"<tr><td>{r.t0:%m-%d}</td><td>{esc(r.symbol)}</td><td>{o['n']}일째</td><td>{'성공' if o['y10'] else ('실패' if o['complete'] else '진행 중')}</td><td>{rr}</td><td>{s1(o['ret'])}</td><td>{s1(o['mae'])}</td></tr>"
rec_tbl = f"<div class=scrollx><table><tr><th>급등일</th><th>종목</th><th>경과</th><th>종가 기준</th><th>장중 도달</th><th>현재</th><th>최대 하락</th></tr>{rrows}</table></div>" if rrows else "<p class=dim>아직 기록이 없습니다.</p>"
CSS = open(os.path.join(HERE, "report_css.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(HERE, "report_css.txt")) else ""
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
HT = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>오늘의 급등 신호</title><style>{CSS}</style></head><body><main>
<p class=dim>미국 {last:%Y-%m-%d} 장 마감 기준 · {now} (한국) 갱신 · 다음 미국 개장 시가 진입 가정</p>
<h1>오늘의 급등 신호</h1>
<p class=warn>투자자문이 아닙니다. 과거 데이터로 만든 규칙의 신호와 통계이며 매수·매도 권유가 아닙니다.</p>
<div class=lead><p>오늘 급등 {len(today)}건 → 황금알 후보 {int(today.cand.sum())}건 → 규칙이 고른 종목 <b>{len(picks)}건</b></p></div>
<h2>1. 오늘 고른 종목</h2>{cards}
<h2>2. 이 규칙의 예상 확률과 목표가</h2>
<p class=dim>규칙을 만든 자료 밖에서 같은 규칙이 고른 급등 {PR['y10']['n']}건(결과가 다 나온 것)의 실제 비율입니다. 매일 갱신됩니다.</p>
<div class=scrollx><table><tr><th>항목</th><th>확률</th><th>90% 범위</th><th>최근 1년</th></tr>{prob_rows}</table></div>
<p><b>목표가 규칙:</b> 1차 = 닿을 확률 70% 이상인 가장 높은 값({('+' + str(int(t1 * 100)) + '%') if t1 else '없음'}), 2차 = 50% 이상인 가장 높은 값({('+' + str(int(t2 * 100)) + '%') if t2 else '없음'})</p>
<h3>매도 방식별 과거 성적 (같은 {PR['y10']['n']}건)</h3>
<div class=scrollx><table><tr><th>매도 방식</th><th>승률</th><th>거래당 평균</th><th>중간값</th><th>최악</th></tr>{ex_rows}</table></div>
<p class=dim>비용 전입니다. C는 같은 날 두 가격이 함께 찍히면 시가에서 가까운 쪽이 먼저라고 가정했습니다.</p>
<h3>연도별</h3><div class=scrollx><table><tr><th>연도</th><th>건수</th><th>종가 기준 성공</th><th>장중 +10%</th><th>B 평균</th></tr>{by_year}</table></div>
<h2>3. 오늘의 다른 후보</h2>{oth_tbl}
<h2>4. 2026-11-05 이후 신호 기록</h2>{rec_tbl}
<h2>5. 로직</h2><div class=card><ol>
<li><b>급등:</b> 하루 +15% 또는 사흘 +30%, 20일 신고가, 거래량 평소 2배 이상, 거래대금 기준 이상.</li>
<li><b>1단계:</b> 황금알 흔적 규칙 178개 중 하나라도 해당.</li>
<li><b>2단계:</b> 2019~2023년으로 학습해 고정한 점수 모델이, 지난 1년 후보 점수의 상위 10% 문턱을 넘으면 선택.</li>
<li><b>진입·판정:</b> 다음 날 시가 진입. 성공 = 20거래일 안에 종가가 이틀 연속 +10%.</li>
<li><b>확률·목표가:</b> 위 2장의 실제 비율. 시도 42 조합이 통과했다면 종목별로 해당 조합을 함께 표시.</li>
</ol></div>
</main></body></html>"""
os.makedirs(OUTD, exist_ok=True); fn = os.path.join(OUTD, f"급등신호_{last:%Y%m%d}.html")
for f in (fn, os.path.join(OUTD, "급등신호_최신.html"), os.path.join(OUTD, "index.html")): open(f, "w", encoding="utf-8").write(HT)
print(fn, "| history", len(H), "| picks today", len(picks))
