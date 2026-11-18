# Rebuild of sec_parse.py (company table) + filings_x19.py (filing table 2017-06..) from the gzip-saved submissions JSON (untrusted, json only).
# usage: sec_parse2.py SEC_DIR PX_DIR WORK_DIR   -> WORK/sec_company.csv, WORK/x19/filings_x19.pkl
import sys, os, gzip, json, numpy as np, pandas as pd
SEC, PX, W = sys.argv[1:4]; os.makedirs(os.path.join(W, "x19"), exist_ok=True)
cal = pd.DatetimeIndex(np.load(PX + "/cal.npy"))
mp = pd.read_csv(SEC + "/cik_map.csv", dtype=str, keep_default_na=False)
def jload(p): return json.loads(gzip.decompress(open(p, "rb").read()))
comp, rows = [], []
for c in sorted(set(int(x) for x in mp.cik if x)):
    p = os.path.join(SEC, "subs", f"CIK{c:010d}.json.gz")
    if not os.path.exists(p): continue
    try: j = jload(p)
    except Exception: continue
    ba = (j.get("addresses") or {}).get("business") or {}
    comp.append(dict(cik=c, entityType=j.get("entityType"), sic=j.get("sic"), sicDescription=j.get("sicDescription"), category=j.get("category"),
                     stateOfIncorporation=j.get("stateOfIncorporation"), biz_state=ba.get("stateOrCountry"), biz_foreign=ba.get("isForeignLocation"),
                     biz_country=ba.get("country"), former_names=json.dumps(j.get("formerNames") or []), fiscalYearEnd=j.get("fiscalYearEnd"), name=j.get("name")))
    blocks = [j.get("filings", {}).get("recent", {})]
    for f in j.get("filings", {}).get("files", []):
        q = os.path.join(SEC, "subs", f["name"] + ".gz")
        if os.path.exists(q) and f.get("filingTo", "9999") >= "2017-06-01":
            try: blocks.append(jload(q))
            except Exception: pass
    for b in blocks:
        n = len(b.get("form", [])); g = lambda k: b.get(k, [""] * n)
        A, FD, AC, FO, IT, PD = g("accessionNumber"), g("filingDate"), g("acceptanceDateTime"), g("form"), g("items"), g("primaryDocument")
        for i in range(n):
            if FD[i] < "2017-06-01" or FO[i] in ("424B2", "FWP"): continue
            rows.append((c, A[i], FO[i], AC[i], IT[i], PD[i]))
pd.DataFrame(comp).to_csv(os.path.join(W, "sec_company.csv"), index=False)
F = pd.DataFrame(rows, columns=["cik", "acc", "form", "accept", "items", "pdoc"]).drop_duplicates(["cik", "acc"])
acc = pd.to_datetime(F.accept, utc=True, errors="coerce").dt.tz_convert("America/New_York")
F["accept_et"] = acc.dt.tz_localize(None); F = F[F.accept_et.notna()].copy(); acc = acc[F.index]
day = acc.dt.normalize().dt.tz_localize(None); mins = (acc.dt.hour * 60 + acc.dt.minute).to_numpy()
pos = cal.searchsorted(day); same = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == day)
F["react_pos"] = np.where(same & (mins >= 16 * 60), pos + 1, pos).astype(np.int32)
F["open_pos"] = np.where(same & (mins >= 9 * 60 + 30), pos + 1, pos).astype(np.int32)
TXT = {"8-K", "8-K/A", "6-K", "6-K/A", "425", "SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A", "SC TO-T", "SC 14D9", "8-K12B", "8-K12G3"}
F["is_txt"] = F.form.isin(TXT); F.loc[~F.is_txt, "pdoc"] = ""
for c in ("form", "items"): F[c] = F[c].astype("category")
F["cik"] = F.cik.astype(np.int32); F = F.drop(columns=["accept"]).reset_index(drop=True)
F.to_pickle(os.path.join(W, "x19", "filings_x19.pkl")); print("companies", len(comp), "filings", len(F), "ciks", F.cik.nunique(), F.accept_et.min(), F.accept_et.max())
