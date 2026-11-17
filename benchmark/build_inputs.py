"""Build model inputs for the benchmark companies from filings and judgments.

    python benchmark/build_inputs.py

Reads:
- benchmark/data/filings/*.json: figures copied from each company's latest
  quarterly report and market data, one record per figure with its source,
  URL and date (collected Nov 2026).
- benchmark/data/judgments.csv: model inputs that no filing states (moat,
  compute exposure, funding dependence...), each with a reason.

Writes:
- benchmark/data/companies_real.csv: one row per company in the model's
  RAW_COLUMNS layout, money in US dollars.
- benchmark/data/companies_sources.csv: one row per company and input,
  saying whether it came from a filing, market data, a judgment, or was
  derived, and from what.

Derivations, stated so they can be challenged:
- monthly_net_burn = -(operating cash flow - capex) / 3 for the latest
  quarter. Negative means the company generates cash after capex.
- revenue_annual = trailing 12 months where available, else 4 x quarter.
- Where only half-year cash flows are published, burn uses those / 6.
- Money in other currencies is converted at the rate in each file's
  fx_to_usd record (local currency per US dollar).
- revenue_growth = latest quarter over the same quarter a year earlier.
- equity_value_now = market cap. equity_value_ref = market cap x (52-week
  high / current price). The model's reference price for a public company
  is "the last offering price"; for long-listed companies that price is
  decades old, so this benchmark uses the 52-week high as the reference.
  Open choice for the model owner.
- debt excludes operating leases. debt_due_24m is the current portion of
  debt where no maturity table was found (an understatement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_pd.features import RAW_COLUMNS  # noqa: E402

DATA = ROOT / "benchmark" / "data"
M = 1_000_000.0

TICKER_TO_ID = {
    "CRWV": "coreweave", "ORCL": "oracle", "META": "meta", "AMZN": "amazon",
    "NVDA": "nvidia", "AMD": "amd", "AVGO": "broadcom", "INTC": "intel", "GLW": "corning",
    "TSM": "tsmc", "2330.TW": "tsmc", "000660.KS": "sk_hynix", "9984.T": "softbank",
    "VST": "vistra",
}

# Vendor 52-week highs that the collected price history contradicts. The
# highest close in that history is used instead (local currency).
HIGH_OVERRIDES = {
    "sk_hynix": (1_868_000.0, "vendor 52-week high 2,987,000 not in the price history; using highest close 1,868,000 (2026-09-21, stockanalysis history)"),
    "corning": (196.0, "vendor 52-week high 271.78 conflicts with Apr-May 2026 52-week-high headlines near $196 and October prices of $124-128; using 196"),
    "softbank": (6_810.0, "vendor 52-week high 9,074 looks stale; using highest close 6,810 (2026-09-09, stockanalysis history)"),
}

# Field name variants used across the collected files.
ALIASES = {
    "ocf_q": ["operating_cash_flow_quarter", "operating_cash_flow_latest_quarter"],
    "capex_q": ["capex_quarter", "capex_latest_quarter"],
    "vol": ["equity_vol_historical", "equity_volatility", "equity_vol_implied"],
}


def _rec(c: dict, *names: str) -> dict | None:
    for n in names:
        r = c.get(n)
        if isinstance(r, dict) and r.get("value") not in (None, "None", ""):
            return r
    return None


def _num(r: dict | None, fx: float = 1.0) -> float | None:
    if r is None:
        return None
    v = float(r["value"])
    unit = str(r.get("unit", "")).lower()
    if "percent" in unit or unit == "pct":
        return v / 100.0
    if unit.endswith("_m"):
        cur = unit.split("_")[0].upper()
        return v * M * (1.0 if cur == "USD" else fx)
    return v


def _fx(c: dict) -> float:
    """US dollars per unit of the reporting currency."""
    fx = c.get("fx_to_usd")
    if not isinstance(fx, dict):
        return 1.0
    if fx.get("rate") not in (None, "None"):
        # Quoted as local currency per US dollar, e.g. 31.88 TWD per USD.
        return 1.0 / float(fx["rate"])
    if fx.get("value") not in (None, "None"):
        return float(fx["value"])
    return 1.0


def _company_id(ticker: str) -> str:
    for t in [ticker, *[p.strip() for p in ticker.split("/")]]:
        if t in TICKER_TO_ID:
            return TICKER_TO_ID[t]
    raise KeyError(f"unknown ticker {ticker!r}")


def load_filings(data: Path = DATA) -> list[dict]:
    out = []
    for path in sorted((data / "filings").glob("*.json")):
        out.extend(json.loads(path.read_text()))
    return out


def build(data: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs for the companies whose filings, judgments and segment map sit in ``data``."""
    seg = pd.read_csv(data / "segment_map.csv").set_index("company_id")
    judg = pd.read_csv(data / "judgments.csv", dtype={"value": str})
    jmap = {(r.company_id, r.field): r for r in judg.itertuples()}

    rows, prov = [], []
    for c in load_filings(data):
        cid = _company_id(c["ticker"])
        fx = _fx(c)
        row: dict = {"company_id": cid, "segment": seg.loc[cid, "segment"], "public": True}

        def put(field, value, kind, recs=(), note=""):
            row[field] = value
            srcs = [r for r in recs if r]
            prov.append(
                {
                    "company_id": cid,
                    "field": field,
                    "value": value,
                    "kind": kind,
                    "source": " | ".join(r.get("source", "") for r in srcs) or None,
                    "url": " | ".join(r.get("url", "") for r in srcs) or None,
                    "date": " | ".join(str(r.get("date", "")) for r in srcs) or None,
                    "note": note or " | ".join(str(r.get("note", "")) for r in srcs if r.get("note")) or None,
                }
            )

        def from_filing(field, *names, kind="filing"):
            r = _rec(c, *names)
            if r is not None:
                put(field, _num(r, fx), kind, [r])
                return True
            return False

        def from_judgment(field, cast=float):
            j = jmap.get((cid, field))
            if j is None:
                raise KeyError(f"{cid}: no filing figure or judgment for {field}")
            put(field, cast(j.value), "judgment", note=j.note if isinstance(j.note, str) and j.note else "judgment")

        from_filing("cash", "cash_and_st_investments")
        from_filing("debt", "total_debt")
        from_filing("debt_due_24m", "debt_due_within_24m")
        from_filing("total_assets", "total_assets")
        from_filing("hard_assets", "ppe_net")
        from_filing("employees", "employees")
        if not from_filing("gross_margin", "gross_margin_latest_quarter"):
            from_judgment("gross_margin")

        rq = _rec(c, "revenue_latest_quarter")
        rpy = _rec(c, "revenue_same_quarter_prior_year")
        rttm = _rec(c, "revenue_ttm")
        if rttm is not None:
            put("revenue_annual", _num(rttm, fx), "filing", [rttm])
        else:
            put("revenue_annual", 4 * _num(rq, fx), "derived", [rq], "4 x latest quarter")
        put("revenue_growth", _num(rq, fx) / _num(rpy, fx) - 1.0, "derived", [rq, rpy], "latest quarter vs same quarter a year earlier")

        ocf, capex = _rec(c, *ALIASES["ocf_q"]), _rec(c, *ALIASES["capex_q"])
        months, period = 3.0, "latest quarter"
        if ocf is None or capex is None:
            # Some issuers (SoftBank) publish only half-year cash flows.
            ocf, capex, months, period = _rec(c, "operating_cash_flow_6m"), _rec(c, "capex_6m"), 6.0, "latest half year"
        burn = -(_num(ocf, fx) - abs(_num(capex, fx))) / months
        put("monthly_net_burn", burn, "derived", [ocf, capex], f"-(operating cash flow - capex) / {months:g}, {period}")

        mcap = _rec(c, "market_cap")
        px, hi = _rec(c, "share_price_current"), _rec(c, "share_price_52w_high")
        now = _num(mcap, fx)
        put("equity_value_now", now, "market", [mcap])
        high, note = float(hi["value"]), "market cap x 52-week high / current price"
        if cid in HIGH_OVERRIDES:
            high, why = HIGH_OVERRIDES[cid]
            note += "; " + why
        ratio = max(high / float(px["value"]), 1.0)
        put("equity_value_ref", now * ratio, "derived", [mcap, px, hi], note)

        if not from_filing("floating_debt_share", "floating_rate_share_of_debt"):
            from_judgment("floating_debt_share")
        if not from_filing("equity_vol", *ALIASES["vol"], kind="market"):
            from_judgment("equity_vol")
        # Concentration: a judgment (which cites the filing's partial figure)
        # wins, because filings rarely give a true top-five share.
        if (cid, "top5_customer_share") in jmap or not from_filing("top5_customer_share", "top5_customer_share"):
            from_judgment("top5_customer_share")

        for field in (
            "margin_path_score", "moat_score", "compute_cost_share", "compute_revenue_share", "backlog_cover",
            "gpu_asset_share", "liquidation_haircut", "funding_dependence", "ig_customer_share",
            "circular_revenue_share",
        ):
            from_judgment(field)
        from_judgment("revenue_stage", cast=str)
        rows.append(row)

    companies = pd.DataFrame(rows)[RAW_COLUMNS]
    return companies, pd.DataFrame(prov)


UNIVERSE = ROOT / "data" / "universe"


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build credit model inputs from filings and judgments.")
    p.add_argument("--universe", action="store_true",
                   help="build the equity universe's extra names (data/universe/) instead of the CDS set")
    a = p.parse_args(argv)
    if a.universe:
        data, names = UNIVERSE, ("credit_inputs.csv", "credit_sources.csv")
    else:
        data, names = DATA, ("companies_real.csv", "companies_sources.csv")
    companies, prov = build(data)
    companies.to_csv(data / names[0], index=False)
    prov.to_csv(data / names[1], index=False)
    print(f"Wrote {len(companies)} companies and {len(prov)} provenance rows")


if __name__ == "__main__":
    main()
