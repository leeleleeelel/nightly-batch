# Daily job (no Claude needed): refresh the free data sources, recompute the attempt-25 features with the same scripts, score with the FROZEN
# model, write the Korean signal page and open it in Chrome. Run on weekdays at 21:00 KST by Windows Task Scheduler.
# usage: python -I -X utf8 daily_run.py
import os, sys, gzip, json, time, shutil, subprocess, datetime, requests, pandas as pd
C2 = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(C2)
R = os.environ.get("RALLY_BASE", RB); D = os.environ.get("RALLY_DATA", R + "/data"); CORE = D + "/core"; W = D + "/work"
OUTD = os.environ.get("RALLY_OUT", R + "/daily"); LOGD = os.environ.get("RALLY_LOGS", OUTD + "/logs"); os.makedirs(LOGD, exist_ok=True)
for d_ in (CORE + "/meta", W + "/x19", D + "/sec_raw", D + "/sec/subs", D + "/sec/xbrl_raw", D + "/finra", D + "/ftd", D + "/th", D + "/raw_yh", D + "/meta"): os.makedirs(d_, exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M"); LOG = open(f"{LOGD}/run_{stamp}.log", "a", encoding="utf-8")
PY = [sys.executable, "-I", "-X", "utf8", "-W", "ignore"]
def log(m): LOG.write(f"[{datetime.datetime.now():%H:%M:%S}] {m}\n"); LOG.flush(); print(m, flush=True)
def run(name, args):
    log(f"start {name}"); t = time.time(); p = subprocess.run(PY + args, stdout=LOG, stderr=subprocess.STDOUT, cwd=RB)
    log(f"end {name} rc={p.returncode} {round(time.time() - t)}s")
    if p.returncode != 0: raise RuntimeError(f"{name} failed (rc {p.returncode})")
def sec_refresh_recent(n_sessions=5):
    E = pd.read_pickle(CORE + "/events_x19.pkl"); cal = pd.DatetimeIndex(pd.to_datetime(__import__("numpy").load(CORE + "/px/cal.npy")))
    recent = set(E[E.t0 >= cal[-n_sessions]].symbol); mp = pd.read_csv(D + "/sec/cik_map.csv", dtype=str, keep_default_na=False)
    ciks = sorted(set(int(c) for s, c in zip(mp.symbol, mp.cik) if c and s in recent))
    s = requests.Session(); s.headers.update({"User-Agent": "rally-research-script/1.0 (personal research)", "Accept-Encoding": "gzip, deflate"}); ok = 0
    for c in ciks:
        try:
            r = s.get(f"https://data.sec.gov/submissions/CIK{c:010d}.json", timeout=60)
            if r.status_code == 200: open(D + f"/sec/subs/CIK{c:010d}.json.gz", "wb").write(gzip.compress(r.content)); ok += 1
        except Exception: pass
        time.sleep(0.3)
    log(f"sec refresh: {ok}/{len(ciks)} recent-surge companies")
try:
    weekly = datetime.date.today().weekday() == 0
    run("universe", [C2 + "/universe.py", D])
    run("yahoo daily bars (full refresh)", [C2 + "/yhd_fetch2.py", D + "/meta/universe.csv", D + "/raw_yh", "refresh"])
    run("finra", [C2 + "/finra_fetch2.py", D + "/meta/universe.csv", D + "/finra"])
    run("threshold lists", [C2 + "/fetch_th2.py", "nasdaq", "none", D + "/th/nasdaq_rows.csv", D + "/th/nasdaq_status.csv"])
    if not os.path.exists(D + "/sec/cik_map.csv") and os.path.exists(RB + "/seed/cik_map.csv"): shutil.copy(RB + "/seed/cik_map.csv", D + "/sec/cik_map.csv")
    if not os.path.exists(D + "/ftd/ftd_fetch_log.csv") and os.path.exists(RB + "/seed/ftd_fetch_log.csv"): shutil.copy(RB + "/seed/ftd_fetch_log.csv", D + "/ftd/ftd_fetch_log.csv")
    if weekly:   # the latest XBRL quarters once a week (data.sec.gov); the ticker->CIK map is kept (www.sec.gov is not reachable from GitHub runners)
        q = pd.Timestamp.today(); recent_q = {f"CY{(q - pd.DateOffset(months=3 * k)).year}Q{(q - pd.DateOffset(months=3 * k)).quarter}" for k in (1, 2)}
        for f in os.listdir(D + "/sec/xbrl_raw"):
            if any(rq in f for rq in recent_q): os.remove(D + "/sec/xbrl_raw/" + f)
    run("ftd", [C2 + "/fetch_ftd2.py", D + "/ftd", D + "/ftd/ftd_fetch_log.csv"])
    run("sec submissions + xbrl (missing only)", [C2 + "/sec_fetch2.py", D + "/meta/universe.csv", D + "/sec"])
    run("panel", [C2 + "/build_panel2.py", D + "/raw_yh", D + "/meta/universe.csv", CORE])
    run("events", [C2 + "/events2.py", CORE + "/px", CORE + "/events_x19.pkl"])
    run("capture", [C2 + "/capture2.py", CORE])
    sec_refresh_recent()
    run("sec tables", [C2 + "/sec_parse2.py", D + "/sec", CORE + "/px", W])
    for a, b in ((D + "/sec/cik_map.csv", D + "/sec_raw/cik_map.csv"), (D + "/sec/cik_map.csv", CORE + "/meta/cik_map.csv"), (W + "/sec_company.csv", CORE + "/meta/sec_company.csv"),
                 (CORE + "/events_x19.pkl", W + "/x19/events_x19.pkl"), (D + "/ftd/ftd_fetch_log.csv", CORE + "/ftd_fetch_log.csv"),
                 (C2 + "/detect.py", W + "/detect.py"), (C2 + "/labeler.py", W + "/labeler.py"), (C2 + "/detect.py", W + "/x19/detect.py"), (C2 + "/labeler.py", W + "/x19/labeler.py")): shutil.copy(a, b)
    run("svr/si/xbrl", [C2 + "/core_extra2.py", D, CORE, W])
    run("feat_a", [C2 + "/feats_a2.py", W, CORE + "/px"]); shutil.copy(W + "/x19/feat_a.pkl", CORE + "/feat_a.pkl")
    run("feat_b", [C2 + "/feats_b2.py", W, CORE + "/px"]); shutil.copy(W + "/x19/feat_b.pkl", CORE + "/feat_b.pkl")
    run("market state", [C2 + "/regime2.py", CORE])
    run("wq", [C2 + "/wq_feats2.py", CORE, "0"])
    run("rel", [C2 + "/rel_feats2.py", CORE])
    if os.path.exists(RB + "/rel_rules_ge.csv"): shutil.copy(RB + "/rel_rules_ge.csv", CORE + "/rel_rules_ge.csv")
    run("ftd feats", [C2 + "/ftd_feats2.py", CORE, D + "/ftd", D + "/th"])
    run("score", [C2 + "/score_daily.py", CORE, RB + "/a25_model.pkl", OUTD + "/scores_latest.pkl"])
    if os.path.exists(RB + "/a42_rules.json"): run("attempt-42 matches", [C2 + "/a42_match.py", CORE, RB + "/a42_rules.json", OUTD + "/scores_latest.pkl", RB + "/a42_matches.json"])
    args = [C2 + "/render_daily.py", CORE, OUTD + "/scores_latest.pkl", OUTD]
    if os.path.exists(RB + "/a42_matches.json"): args.append(RB + "/a42_matches.json")
    run("page", args)
    page = OUTD + "/급등신호_최신.html"
    if os.path.exists(OUTD + "/업데이트_실패.html"): os.remove(OUTD + "/업데이트_실패.html")
except Exception as e:
    import html, re
    log(f"FAILED: {e}"); LOG.flush(); os.makedirs(OUTD, exist_ok=True); now = datetime.datetime.now(); why = html.escape(str(e))[:300]
    try: print("---- last log lines ----\n" + "".join(open(f"{LOGD}/run_{stamp}.log", encoding="utf-8", errors="replace").readlines()[-40:]), flush=True)
    except Exception: pass
    note = (f"<!--fail-banner--><div style='background:#fde2e1;color:#7a1712;padding:10px 14px;font-family:sans-serif;border-bottom:2px solid #c0392b'>"
            f"⚠ {now:%Y-%m-%d %H:%M} 업데이트 실패: {why}<br>아래는 마지막으로 성공한 업데이트의 내용입니다.</div><!--/fail-banner-->")
    kept = False
    for nm in ("index.html", "급등신호_최신.html"):   # keep the last good page (restored by the workflow) and put a notice on top
        f_ = os.path.join(OUTD, nm)
        if os.path.exists(f_):
            t_ = re.sub(r"<!--fail-banner-->.*?<!--/fail-banner-->", "", open(f_, encoding="utf-8").read(), flags=re.S); m_ = re.search(r"<body[^>]*>", t_)
            open(f_, "w", encoding="utf-8").write(t_[:m_.end()] + note + t_[m_.end():] if m_ else note + t_); kept = True
    fail = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><meta name=robots content=noindex><title>업데이트 실패</title>"
            f"<body style='font-family:sans-serif;padding:1.5em'><h1>업데이트 실패</h1><p>{now:%Y-%m-%d %H:%M}</p><p>{why}</p></body>")
    open(OUTD + "/업데이트_실패.html", "w", encoding="utf-8").write(fail)
    if not kept: open(OUTD + "/index.html", "w", encoding="utf-8").write(fail)
    page = OUTD + ("/급등신호_최신.html" if kept else "/업데이트_실패.html")
if os.name == "nt" and not os.environ.get("RALLY_NO_BROWSER"):
    subprocess.Popen(["cmd", "/c", "start", "", "chrome", page.replace("/", "\\")], shell=False); log("opened " + page)
else:
    log("page ready " + page)
