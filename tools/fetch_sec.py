#!/usr/bin/env python3
"""Download the SEC filings (and daily prices) step 5 needs. Run it on your own computer.

    python tools/fetch_sec.py --user-agent "Your Name your@email.com"

The SEC asks every automated client to say who it is in the User-Agent
header, as a name and an email address; it blocks clients that don't. The
script refuses to run without one, and never stores it.

What it fetches, for the real companies in data/real/universe.csv (build =
xbrl) and the boom firms in data/analogs/panel_firms.csv:

1. The SEC ticker list, to turn tickers into CIK numbers. Boom firms that
   are no longer listed are looked up by name in the SEC's company name
   list instead. Any it can't match are written to unresolved.csv; add the
   CIK by hand to data/analogs/cik_overrides.csv and run again.
2. Each company's XBRL "company facts", cut down to the concepts the model
   reads (ai_pd/xbrl.py, CONCEPTS).
3. Each company's filing index (forms, dates, accession numbers, 8-K item
   codes), which is where bankruptcies (8-K item 1.03) and delistings show.
4. Daily prices from Yahoo's chart endpoint, falling back to Stooq, unless
   you pass --no-prices. Many failed boom firms have no free price history;
   that is expected and is reported, not an error.

Everything lands in one folder (sec_pull/ by default) with a manifest of
every URL, the time it was fetched and a SHA-256 of what was saved. Zip
that folder and attach it to the step 5 thread in Claude Duet.

Uses only the Python standard library (3.9+). Stays under the SEC's limit
of 10 requests a second. A full run takes roughly 10 to 20 minutes. It can
be stopped and re-run: anything already saved is skipped unless --refresh.

    --dry-run       print the URLs it would fetch and stop
    --only ID ...   fetch just these company or firm ids
    --out DIR       output folder (default sec_pull)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "real" / "universe.csv"
PANEL = ROOT / "data" / "analogs" / "panel_firms.csv"
OVERRIDES = ROOT / "data" / "analogs" / "cik_overrides.csv"

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
NAMES_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
             "?period1={p1}&period2={p2}&interval=1d&events=split")
STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}.us&i=d"

KEEP_FORMS = {"10-K", "10-K/A", "10-KT", "10-Q", "10-Q/A", "8-K", "8-K/A", "20-F", "40-F",
              "15-12B", "15-12G", "15-15D", "25", "25-NSE", "S-4", "DEFM14A", "SC TO-T"}
PRICE_START = "2009-01-01"
SEC_GAP = 0.12  # seconds between SEC requests (under 10 a second)
MAX_TRIES = 4


def load_concepts() -> set[tuple[str, str]]:
    """The concept whitelist from ai_pd/xbrl.py, loaded by path so pandas isn't needed."""
    spec = importlib.util.spec_from_file_location("_xbrl", ROOT / "ai_pd" / "xbrl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_xbrl"] = mod
    spec.loader.exec_module(mod)
    return mod.concept_whitelist()


def norm_name(s: str) -> str:
    s = s.upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    drop = {"INC", "CORP", "CORPORATION", "CO", "LTD", "LLC", "PLC", "HOLDINGS", "HOLDING",
            "COMPANY", "THE", "SA", "NV", "LP", "GROUP", "INTERNATIONAL", "INTL"}
    return " ".join(w for w in s.split() if w not in drop)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class Fetcher:
    def __init__(self, user_agent: str, out: Path, refresh: bool):
        self.ua, self.out, self.refresh = user_agent, out, refresh
        self.manifest: list[dict] = []
        self._last_sec = 0.0

    def get(self, url: str, sec: bool = True) -> bytes | None:
        headers = {"Accept-Encoding": "gzip"}
        headers["User-Agent"] = self.ua if sec else "Mozilla/5.0 (research script)"
        for attempt in range(MAX_TRIES):
            if sec:
                wait = SEC_GAP - (time.monotonic() - self._last_sec)
                if wait > 0:
                    time.sleep(wait)
                self._last_sec = time.monotonic()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                    body = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code in (403, 429, 500, 502, 503) and attempt < MAX_TRIES - 1:
                    time.sleep(2 ** attempt * (5 if e.code in (403, 429) else 1))
                    continue
                print(f"  HTTP {e.code} for {url}", file=sys.stderr)
                return None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < MAX_TRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  failed {url}: {e}", file=sys.stderr)
                return None
        return None

    def save(self, rel: str, data: bytes, url: str, kind: str, key: str) -> None:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self.manifest.append({"kind": kind, "id": key, "path": rel, "url": url,
                              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})

    def have(self, rel: str) -> bool:
        return (self.out / rel).exists() and not self.refresh


def reduce_facts(raw: dict, keep: set[tuple[str, str]]) -> dict:
    facts = {}
    for tax, concepts in raw.get("facts", {}).items():
        for name, body in concepts.items():
            if (tax, name) in keep:
                facts.setdefault(tax, {})[name] = {"units": body.get("units", {})}
    return {"cik": raw.get("cik"), "entityName": raw.get("entityName"), "facts": facts}


def reduce_filings(block: dict) -> list[dict]:
    cols = ["form", "filingDate", "reportDate", "accessionNumber", "items"]
    n = len(block.get("form", []))
    rows = []
    for i in range(n):
        if block["form"][i] in KEEP_FORMS:
            rows.append({c: (block.get(c) or [""] * n)[i] for c in cols})
    return rows


def fetch_submissions(f: Fetcher, cik: int) -> dict | None:
    body = f.get(SUBS_URL.format(cik=cik))
    if body is None:
        return None
    raw = json.loads(body)
    filings = reduce_filings(raw.get("filings", {}).get("recent", {}))
    urls = [SUBS_URL.format(cik=cik)]
    for page in raw.get("filings", {}).get("files", []):
        url = SUBS_PAGE_URL.format(name=page["name"])
        pb = f.get(url)
        if pb is not None:
            filings += reduce_filings(json.loads(pb))
            urls.append(url)
    return {"cik": cik, "name": raw.get("name"), "sic": raw.get("sic"),
            "sicDescription": raw.get("sicDescription"), "tickers": raw.get("tickers"),
            "exchanges": raw.get("exchanges"), "formerNames": raw.get("formerNames"),
            "filings": sorted(filings, key=lambda r: r["filingDate"]), "source_urls": urls}


def yahoo_prices(f: Fetcher, ticker: str) -> tuple[str, bytes] | None:
    p1 = int(datetime.fromisoformat(PRICE_START).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(time.time())
    url = YAHOO_URL.format(ticker=urllib.parse.quote(ticker), p1=p1, p2=p2)
    body = f.get(url, sec=False)
    if not body:
        return None
    try:
        res = json.loads(body)["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["date", "close", "adj_close", "volume"])
    for t, c, a, v in zip(ts, q["close"], adj, q.get("volume", [None] * len(ts))):
        if c is not None:
            w.writerow([datetime.fromtimestamp(t, timezone.utc).date().isoformat(), c, a, v])
    return url, out.getvalue().encode()


def stooq_prices(f: Fetcher, ticker: str) -> tuple[str, bytes] | None:
    url = STOOQ_URL.format(ticker=ticker.lower().replace(".", "-"))
    body = f.get(url, sec=False)
    if not body or not body.startswith(b"Date,"):
        return None
    rows = list(csv.reader(io.StringIO(body.decode())))
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["date", "close", "adj_close", "volume"])
    for r in rows[1:]:
        if len(r) >= 5 and r[0] >= PRICE_START:
            w.writerow([r[0], r[4], r[4], r[5] if len(r) > 5 else ""])
    return url, out.getvalue().encode()


def targets(only: set[str]) -> list[dict]:
    rows = []
    for r in read_csv(UNIVERSE):
        if r["build"] == "xbrl":
            rows.append({"who": "universe", "id": r["company_id"], "ticker": r["ticker"], "name": r["name"]})
    for r in read_csv(PANEL):
        rows.append({"who": "analog", "id": r["firm_id"], "ticker": r["ticker_then"], "name": r["name"]})
    return [r for r in rows if not only or r["id"] in only]


def resolve(f: Fetcher, rows: list[dict], dry: bool) -> None:
    overrides = {r["firm_id"]: int(r["cik"]) for r in read_csv(OVERRIDES) if r.get("cik")}
    tick = {}
    if not dry:
        body = f.get(TICKERS_URL)
        if body is None:
            sys.exit("Could not download the SEC ticker list. Check the --user-agent value and your connection.")
        f.save("ref/company_tickers.json", body, TICKERS_URL, "reference", "tickers")
        tick = {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(body).values()}
    names_needed = []
    for r in rows:
        if r["id"] in overrides:
            r["cik"], r["how"] = overrides[r["id"]], "override"
        elif r["ticker"] and r["ticker"].upper() in tick and r["who"] == "universe":
            r["cik"], r["how"] = tick[r["ticker"].upper()], "ticker"
        else:
            names_needed.append(r)
    if dry:
        return
    # Boom firms: tickers are often reused by other companies years later, so
    # match on name first and only fall back to today's ticker list.
    if names_needed:
        print(f"Looking up {len(names_needed)} names in the SEC company list (a large file)...")
        body = f.get(NAMES_URL)
        index: dict[str, set[int]] = {}
        if body is not None:
            f.save("ref/cik-lookup-data.sha256", hashlib.sha256(body).hexdigest().encode(),
                   NAMES_URL, "reference", "names")
            for line in body.decode("latin-1").splitlines():
                parts = line.rstrip(":").rsplit(":", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    index.setdefault(norm_name(parts[0]), set()).add(int(parts[1]))
        for r in names_needed:
            hits = index.get(norm_name(r["name"]), set())
            if len(hits) == 1:
                r["cik"], r["how"] = hits.pop(), "name"
            elif r["ticker"] and r["ticker"].upper() in tick:
                r["cik"], r["how"] = tick[r["ticker"].upper()], "ticker (check: tickers get reused)"
            else:
                r["how"] = f"unresolved ({len(hits)} name matches)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--user-agent", help='your name and email, e.g. "Jane Doe jane@example.com"')
    ap.add_argument("--out", default="sec_pull")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--no-prices", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if not a.dry_run and (not a.user_agent or "@" not in a.user_agent or " " not in a.user_agent.strip()):
        print('The SEC requires a User-Agent with your name and email, for example:\n'
              '  python tools/fetch_sec.py --user-agent "Jane Doe jane@example.com"', file=sys.stderr)
        return 2

    out = Path(a.out)
    f = Fetcher(a.user_agent or "dry-run", out, a.refresh)
    rows = targets(set(a.only))
    resolve(f, rows, a.dry_run)

    if a.dry_run:
        print(f"GET {TICKERS_URL}")
        if any("cik" not in r for r in rows):
            print(f"GET {NAMES_URL}")
        for r in rows:
            c = r.get("cik")
            label = f"{r['who']}:{r['id']}"
            if c:
                print(f"GET {FACTS_URL.format(cik=c)}  # {label}")
                print(f"GET {SUBS_URL.format(cik=c)}  # {label}")
            else:
                print(f"GET (CIK resolved at run time) companyfacts + submissions  # {label}")
            if not a.no_prices and r["ticker"]:
                print(f"GET {YAHOO_URL.format(ticker=r['ticker'], p1='<2009-01-01>', p2='<now>')}  # {label}")
        return 0

    keep = load_concepts()
    out.mkdir(parents=True, exist_ok=True)
    done_cik: set[int] = set()
    for i, r in enumerate(rows, 1):
        c = r.get("cik")
        print(f"[{i}/{len(rows)}] {r['who']} {r['id']} {r['ticker']} cik={c or '-'} ({r.get('how', '')})")
        if c and c not in done_cik:
            done_cik.add(c)
            rel = f"facts/{c}.json"
            if not f.have(rel):
                url = FACTS_URL.format(cik=c)
                body = f.get(url)
                r["facts"] = body is not None
                if body is not None:
                    f.save(rel, json.dumps(reduce_facts(json.loads(body), keep)).encode(), url, "facts", r["id"])
            else:
                r["facts"] = True
            rel = f"submissions/{c}.json"
            if not f.have(rel):
                subs = fetch_submissions(f, c)
                if subs is not None:
                    f.save(rel, json.dumps(subs).encode(), subs["source_urls"][0], "submissions", r["id"])
        if not a.no_prices and r["ticker"]:
            rel = f"prices/{r['who']}/{r['id']}.csv"
            if f.have(rel):
                r["prices"] = "kept"
            else:
                got = yahoo_prices(f, r["ticker"]) or stooq_prices(f, r["ticker"])
                r["prices"] = "none" if got is None else ("yahoo" if "yahoo" in got[0] else "stooq")
                if got is not None:
                    f.save(rel, got[1], got[0], "prices", r["id"])
                time.sleep(0.5)

    cols = ["who", "id", "ticker", "name", "cik", "how", "facts", "prices"]
    with (out / "cik_map.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    unresolved = [r for r in rows if not r.get("cik")]
    with (out / "unresolved.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["who", "id", "ticker", "name", "how"], extrasaction="ignore")
        w.writeheader()
        w.writerows(unresolved)
    old = read_csv(out / "manifest.csv") if not a.refresh else []
    paths = {m["path"] for m in f.manifest}
    manifest = [m for m in old if m["path"] not in paths] + f.manifest
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "id", "path", "url", "fetched_at", "sha256", "bytes"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nDone: {len(rows) - len(unresolved)} of {len(rows)} matched to a CIK, "
          f"{len(unresolved)} unresolved (see {out}/unresolved.csv).")
    if not a.no_prices:
        print(f"Prices found for {sum(1 for r in rows if r.get('prices') not in (None, 'none'))} of {len(rows)}.")
    print(f"Now zip the '{out}' folder and attach the zip to the step 5 thread.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
