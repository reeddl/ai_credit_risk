"""Credit model inputs for the real companies in data/real/universe.csv.

    python -m ai_pd.real --pull sec_pull --as-of 2026-11-14

Reads an SEC pull made by ``tools/fetch_sec.py`` and writes
``data/real/companies.csv`` (RAW_COLUMNS, US dollars) and
``data/real/companies_sources.csv`` (one row per company and input, in the
benchmark's provenance format).

Where each input comes from:

- Financial fields: SEC XBRL company facts as they stood on the as-of date
  (``ai_pd.xbrl``).
- Market fields: daily closes from the pull and shares outstanding from the
  filings. equity_value_now = shares x last close; equity_value_ref =
  shares x highest close in the past year (the benchmark's convention);
  equity_vol = annualised standard deviation of daily log returns over the
  past year. With no prices, these stay missing and the Merton cross-check
  is skipped for that company.
- Judgment fields (moat, compute exposure, customers...): the benchmark's
  hand judgments where a company has them, then data/universe/judgments.csv,
  then the segment's default from data/real/segment_defaults.csv. Every
  segment default is marked ``kind = segment_default`` so it can be found
  and replaced.
- Companies marked ``build = hand`` (TSMC, SK Hynix, SoftBank: IFRS or
  non-SEC filers) are copied from the benchmark set unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .features import RAW_COLUMNS
from .xbrl import CompanyFacts, financial_inputs, shares_outstanding

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"
BENCH = ROOT / "benchmark" / "data"
UNIVERSE_JUDGMENTS = ROOT / "data" / "universe" / "judgments.csv"

FINANCIAL = ["cash", "monthly_net_burn", "revenue_annual", "revenue_growth", "gross_margin",
             "employees", "debt", "debt_due_24m", "total_assets", "hard_assets"]
MARKET = ["equity_value_ref", "equity_value_now", "equity_vol"]
JUDGMENT = [c for c in RAW_COLUMNS if c not in FINANCIAL + MARKET + ["company_id", "segment", "public"]]


def load_universe(path: Path = REAL / "universe.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def cik_map(pull: Path) -> dict[tuple[str, str], int]:
    m = pd.read_csv(pull / "cik_map.csv", dtype={"cik": "Int64"})
    return {(r.who, r.id): int(r.cik) for r in m.itertuples() if pd.notna(r.cik)}


def load_prices(pull: Path, who: str, cid: str) -> pd.Series | None:
    p = pull / "prices" / who / f"{cid}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).dropna(subset=["close"])
    if df.empty:
        return None
    return df.set_index("date")["close"].astype(float).sort_index()


def market_fields(closes: pd.Series | None, shares: float | None, as_of: date) -> dict:
    out = {k: float("nan") for k in MARKET}
    if closes is None or not shares:
        return out
    year = closes[(closes.index > pd.Timestamp(as_of - timedelta(days=365)))
                  & (closes.index <= pd.Timestamp(as_of))]
    if len(year) < 60:
        return out
    rets = np.log(year).diff().dropna()
    out["equity_value_now"] = shares * float(year.iloc[-1])
    out["equity_value_ref"] = shares * float(year.max())
    out["equity_vol"] = float(rets.std() * math.sqrt(252))
    return out


def judgment_table() -> dict[tuple[str, str], tuple[object, str, str]]:
    """(company_id, field) -> (value, source, note) from hand judgments."""
    out = {}
    for path, label in [(UNIVERSE_JUDGMENTS, "data/universe/judgments.csv"),
                        (BENCH / "judgments.csv", "benchmark/data/judgments.csv")]:
        if path.exists():
            for r in pd.read_csv(path).itertuples():
                out[(r.company_id, r.field)] = (r.value, label, r.note)
    return out


def _coerce(field: str, value):
    if field == "revenue_stage":
        return str(value)
    return float(value)


def build_company(row, cf: CompanyFacts, closes, as_of: date, judgments, defaults) -> tuple[dict, list[dict], list[str]]:
    built = financial_inputs(cf, as_of, row.company_id)
    rec = {"company_id": row.company_id, "segment": row.segment, "public": True}
    for f in FINANCIAL:
        rec[f] = built.values.get(f, float("nan"))
    sources = list(built.sources)

    sh = shares_outstanding(cf, as_of)
    mkt = market_fields(closes, sh[0] if sh else None, as_of)
    rec.update(mkt)
    for f in MARKET:
        sources.append({"company_id": row.company_id, "field": f, "value": mkt[f],
                        "kind": "market" if not math.isnan(mkt[f]) else "missing",
                        "source": "daily closes in the SEC pull; shares from SEC XBRL" if sh else None,
                        "url": sh[1].link(cf.cik) if sh else None,
                        "date": as_of.isoformat(),
                        "note": {"equity_value_now": "shares x last close",
                                 "equity_value_ref": "shares x highest close in the past year",
                                 "equity_vol": "annualised std of daily log returns, past year"}[f]})

    # A figure the filings or prices don't give falls back to a hand judgment
    # where one exists (gross margin for companies that don't report it, say).
    for f in FINANCIAL + MARKET:
        if isinstance(rec[f], float) and math.isnan(rec[f]) and (row.company_id, f) in judgments:
            v, src, note = judgments[(row.company_id, f)]
            rec[f] = float(v)
            sources = [s for s in sources if s["field"] != f]
            sources.append({"company_id": row.company_id, "field": f, "value": rec[f], "kind": "judgment",
                            "source": src, "url": None, "date": None, "note": note})

    seg = defaults.loc[row.segment]
    for f in JUDGMENT:
        if (row.company_id, f) in judgments:
            v, src, note = judgments[(row.company_id, f)]
            kind = "judgment"
        else:
            v, src, note, kind = seg[f], "data/real/segment_defaults.csv", f"default for {row.segment}", "segment_default"
        rec[f] = _coerce(f, v)
        sources.append({"company_id": row.company_id, "field": f, "value": rec[f], "kind": kind,
                        "source": src, "url": None, "date": None, "note": note})
    problems = list(built.problems) + ([] if sh else ["no shares outstanding"]) + \
        (["no price history"] if closes is None else [])
    return rec, sources, problems


def build(pull: Path, as_of: date, universe: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (companies, sources, build report)."""
    universe = load_universe() if universe is None else universe
    ciks = cik_map(pull)
    judgments = judgment_table()
    defaults = pd.read_csv(REAL / "segment_defaults.csv").set_index("segment")
    bench = pd.read_csv(BENCH / "companies_real.csv").set_index("company_id")
    bench_src = pd.read_csv(BENCH / "companies_sources.csv")

    rows, sources, report = [], [], []
    for row in universe.itertuples():
        if row.build == "hand":
            rows.append(bench.loc[row.company_id].to_dict() | {"company_id": row.company_id})
            sources += bench_src[bench_src.company_id == row.company_id].to_dict("records")
            report.append({"company_id": row.company_id, "status": "copied from benchmark", "problems": ""})
            continue
        cik = ciks.get(("universe", row.company_id))
        fpath = pull / "facts" / f"{cik}.json"
        if cik is None or not fpath.exists():
            report.append({"company_id": row.company_id, "status": "not built", "problems": "no company facts in the pull"})
            continue
        cf = CompanyFacts.load(fpath)
        if not cf.has_usd_gaap():
            report.append({"company_id": row.company_id, "status": "not built", "problems": "no US GAAP facts in US dollars"})
            continue
        rec, src, problems = build_company(row, cf, load_prices(pull, "universe", row.company_id),
                                           as_of, judgments, defaults)
        rows.append(rec)
        sources += src
        report.append({"company_id": row.company_id, "status": "built", "problems": "; ".join(problems)})

    companies = pd.DataFrame(rows)[RAW_COLUMNS]
    return companies, pd.DataFrame(sources), pd.DataFrame(report)


def compare_to_benchmark(companies: pd.DataFrame) -> pd.DataFrame:
    """Built financial fields next to the hand-built benchmark rows, with the ratio."""
    bench = pd.read_csv(BENCH / "companies_real.csv").set_index("company_id")
    got = companies.set_index("company_id")
    out = []
    for cid in bench.index.intersection(got.index):
        for f in FINANCIAL:
            b, g = float(bench.at[cid, f]), float(got.at[cid, f])
            out.append({"company_id": cid, "field": f, "benchmark": b, "built": g,
                        "ratio": g / b if b not in (0.0,) and not math.isnan(b) else float("nan")})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build real-company credit inputs from an SEC pull.")
    ap.add_argument("--pull", type=Path, required=True)
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--out", type=Path, default=REAL)
    a = ap.parse_args(argv)
    companies, sources, report = build(a.pull, date.fromisoformat(a.as_of))
    a.out.mkdir(parents=True, exist_ok=True)
    companies.to_csv(a.out / "companies.csv", index=False)
    sources.to_csv(a.out / "companies_sources.csv", index=False)
    report.to_csv(a.out / "build_report.csv", index=False)
    compare_to_benchmark(companies).to_csv(a.out / "benchmark_check.csv", index=False)
    print(f"Built {len(companies)} companies; report in {a.out / 'build_report.csv'}")


if __name__ == "__main__":
    main()
