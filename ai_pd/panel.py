"""The post-2009 boom panel: firm-years with point-in-time inputs and default labels.

    python -m ai_pd.panel --pull sec_pull

One observation per 10-K a boom firm filed inside its boom's window
(data/analogs/booms.csv). Inputs are the firm's XBRL facts as they stood on
that filing date. The label asks whether the firm defaulted within the next
12 months, under the model's default definition:

- **bankruptcy**: an 8-K reporting item 1.03 (bankruptcy or receivership),
  read from the firm's SEC filing index;
- **deep loss**: the share price closed at or below 10% of its close on the
  observation date within the next 12 months (where the pull has prices).

Missed payments and shutdowns without a bankruptcy filing show up only
through the price test. A firm-year with no default and nothing to show the
firm was still followed a year later (no filing after about ten months and
no prices past twelve) is kept in the panel file but left out of the fit,
because a quiet failure there would be read as survival.

The ``expected`` column in data/analogs/panel_firms.csv is written from
memory and is never used as a label; build_report.csv sets it beside what
the filings and prices say, so mismatches can be checked by hand.

Only financial inputs are built here. The judgment inputs (moat, compute
exposure, customers...) have no history, so the fit leaves them at their
hand weights.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .features import RAW_COLUMNS
from .real import cik_map, load_prices
from .xbrl import CompanyFacts, financial_inputs

ROOT = Path(__file__).resolve().parent.parent
ANALOGS = ROOT / "data" / "analogs"
OUT = ANALOGS / "credit_panel"
HORIZON = 365
FOLLOW_DAYS = 300
MARKDOWN = 0.90

# Judgment inputs held at fixed values in the panel. Their features are not
# fitted; these only let the scorecard code run.
PANEL_JUDGMENTS = {
    "revenue_stage": "recurring", "margin_path_score": 3.0, "moat_score": 3.0,
    "compute_cost_share": 0.0, "compute_revenue_share": 0.0, "backlog_cover": 0.0,
    "gpu_asset_share": 0.0, "floating_debt_share": 0.25, "liquidation_haircut": 0.5,
    "funding_dependence": 0.3, "top5_customer_share": float("nan"),
    "ig_customer_share": float("nan"), "circular_revenue_share": 0.0,
}


def load_booms(path: Path = ANALOGS / "booms.csv") -> pd.DataFrame:
    b = pd.read_csv(path)
    # A 10-K for fiscal year Y is filed early in Y+1.
    b["start"] = pd.to_datetime(b["first_fy"].astype(str) + "-07-01")
    b["end"] = pd.to_datetime((b["last_fy"] + 1).astype(str) + "-06-30")
    return b.set_index("boom")


def load_submissions(pull: Path, cik: int) -> dict | None:
    p = pull / "submissions" / f"{cik}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bankruptcy_dates(subs: dict) -> list[date]:
    out = []
    for f in subs.get("filings", []):
        if f["form"].startswith("8-K") and "1.03" in str(f.get("items", "")).split(","):
            out.append(date.fromisoformat(f["filingDate"]))
    return sorted(out)


def label(t: date, subs: dict, closes: pd.Series | None) -> dict:
    """Default within 12 months of ``t``, and how it was found."""
    end = t + timedelta(days=HORIZON)
    bk = [d for d in bankruptcy_dates(subs) if t < d <= end]
    out = {"bankruptcy": bool(bk), "bankruptcy_date": bk[0].isoformat() if bk else None,
           "deep_loss": False, "max_drawdown_12m": float("nan"), "has_prices": False}
    if closes is not None:
        start = closes[closes.index <= pd.Timestamp(t)]
        window = closes[(closes.index > pd.Timestamp(t)) & (closes.index <= pd.Timestamp(end))]
        if len(start) and len(window):
            p0 = float(start.iloc[-1])
            dd = float(window.min()) / p0 - 1.0
            out.update(has_prices=True, max_drawdown_12m=dd, deep_loss=dd <= -MARKDOWN)
    later = [date.fromisoformat(f["filingDate"]) for f in subs.get("filings", [])]
    followed = any((d - t).days >= FOLLOW_DAYS for d in later)
    if closes is not None and len(closes) and closes.index.max() >= pd.Timestamp(end):
        followed = True
    out["default"] = out["bankruptcy"] or out["deep_loss"]
    out["followed"] = followed or out["default"]
    return out


def build(pull: Path, firms: pd.DataFrame | None = None, booms: pd.DataFrame | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (firm-years, per-firm build report)."""
    firms = pd.read_csv(ANALOGS / "panel_firms.csv") if firms is None else firms
    booms = load_booms() if booms is None else booms
    ciks = cik_map(pull)
    rows, report = [], []
    for firm in firms.itertuples():
        cik = ciks.get(("analog", firm.firm_id))
        rep = {"firm_id": firm.firm_id, "boom": firm.boom, "expected": firm.expected, "cik": cik,
               "firm_years": 0, "defaults": 0, "problems": ""}
        report.append(rep)
        fpath = pull / "facts" / f"{cik}.json"
        subs = load_submissions(pull, cik) if cik else None
        if cik is None or subs is None:
            rep["problems"] = "no CIK or no filing index in the pull"
            continue
        cf = CompanyFacts.load(fpath) if fpath.exists() else None
        closes = load_prices(pull, "analog", firm.firm_id)
        b = booms.loc[firm.boom]
        tenks = sorted({date.fromisoformat(f["filingDate"]) for f in subs["filings"]
                        if f["form"] in ("10-K", "10-KT")
                        and b.start <= pd.Timestamp(f["filingDate"]) <= b.end})
        probs = set()
        for t in tenks:
            built = financial_inputs(cf, t, firm.firm_id) if cf is not None else None
            if built is None or not built.values:
                probs.update(built.problems if built else ["no XBRL company facts"])
                continue
            probs.update(built.problems)
            rec = {"firm_id": firm.firm_id, "boom": firm.boom, "as_of": t.isoformat(),
                   "period_end": built.period_end.isoformat() if built.period_end else None}
            rec.update(built.values)
            rec.update(label(t, subs, closes))
            rows.append(rec)
            rep["firm_years"] += 1
            rep["defaults"] += int(rec["default"])
        if not tenks:
            probs.add("no 10-K inside the boom window")
        rep["has_prices"] = closes is not None
        rep["bankruptcy_8k"] = ", ".join(d.isoformat() for d in bankruptcy_dates(subs)) or None
        rep["problems"] = "; ".join(sorted(probs))
    return pd.DataFrame(rows), pd.DataFrame(report)


def to_raw(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel rows in the scorecard's RAW_COLUMNS layout (judgment inputs fixed)."""
    df = panel.copy()
    df["company_id"] = df["firm_id"] + "@" + df["as_of"]
    df["segment"] = "infra_software"  # any valid segment; only feature scores are used
    df["public"] = True
    for k, v in PANEL_JUDGMENTS.items():
        df[k] = v
    for k in ["equity_value_ref", "equity_value_now", "equity_vol"]:
        df[k] = np.nan
    for c in RAW_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[RAW_COLUMNS]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the boom panel from an SEC pull.")
    ap.add_argument("--pull", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    panel, report = build(a.pull)
    a.out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(a.out / "firm_years.csv", index=False)
    report.to_csv(a.out / "build_report.csv", index=False)
    used = panel[panel["followed"]] if len(panel) else panel
    print(f"{len(panel)} firm-years from {report.firm_years.gt(0).sum()} firms; "
          f"{int(used['default'].sum()) if len(used) else 0} defaults among {len(used)} followed firm-years.")


if __name__ == "__main__":
    main()
