# Universe of US-listed common stocks / ordinary shares / ADRs from the NasdaqTrader symbol directory (as the original eligible.csv).
# Files are untrusted text, parsed with pandas only. usage: universe.py DATA_DIR
import sys, os, io, requests, pandas as pd
D = sys.argv[1]; s = requests.Session(); s.headers.update({"User-Agent": "rally-research-script/1.0 (personal research)"})
def get(name):
    r = s.get(f"https://www.nasdaqtrader.com/dynamic/SymDir/{name}", timeout=60); r.raise_for_status()
    open(os.path.join(D, "meta", name), "wb").write(r.content); return pd.read_csv(io.BytesIO(r.content), sep="|", dtype=str, keep_default_na=False)
nl, ol = get("nasdaqlisted.txt"), get("otherlisted.txt")
nl = nl[~nl.Symbol.str.startswith("File Creation")]; ol = ol[~ol["ACT Symbol"].str.startswith("File Creation")]
BAD = r"(?i)\b(warrants?|rights?|units?|preferred|notes?|debentures?|subordinated|depositary shares? representing|% |etn|exchange traded|trust certificates?)\b"
a = nl[(nl["Test Issue"] == "N") & (nl.ETF == "N")].assign(symbol=lambda x: x.Symbol, exch="NASDAQ", name=lambda x: x["Security Name"])
b = ol[(ol["Test Issue"] == "N") & (ol.ETF == "N") & ol.Exchange.isin(["N", "A", "P", "Z"])].assign(symbol=lambda x: x["ACT Symbol"], exch=lambda x: x.Exchange, name=lambda x: x["Security Name"])
U = pd.concat([a[["symbol", "exch", "name"]], b[["symbol", "exch", "name"]]], ignore_index=True)
U = U[~U.symbol.str.contains(r"[\$\^]") & ~U.name.str.contains(BAD, regex=True)].drop_duplicates("symbol")
U["yahoo"] = U.symbol.str.replace(".", "-", regex=False)
U.sort_values("symbol").to_csv(os.path.join(D, "meta", "universe.csv"), index=False)
print("universe", len(U), U.exch.value_counts().to_dict())
