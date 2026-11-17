"""A fictional company in the SEC's companyfacts and submissions schema, and a fake pull.

"Example Compute Corp" (CIK 9999901) has a December year end. Revenue by
quarter is 100, 110, 120, 130 in 2025 and 150, 160 (restated to 165), 180 in
2026. Like most filers, 10-Qs report three-month and year-to-date revenue,
and cash flows only year to date. Figures are in dollars.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CIK = 9999901
Q = {"2025-03-31": ("2025-01-01", 100), "2025-06-30": ("2025-04-01", 110),
     "2025-09-30": ("2025-07-01", 120), "2025-12-31": ("2025-10-01", 130),
     "2026-03-31": ("2026-01-01", 150), "2026-06-30": ("2026-04-01", 165),
     "2026-09-30": ("2026-07-01", 180)}
FILED = {"2025-03-31": "2025-05-05", "2025-06-30": "2025-08-05", "2025-09-30": "2025-11-05",
         "2025-12-31": "2026-02-20", "2026-03-31": "2026-05-05", "2026-06-30": "2026-08-05",
         "2026-09-30": "2026-11-05"}
FORM = {k: ("10-K" if k.endswith("12-31") else "10-Q") for k in FILED}


def _f(start, end, val, filed, form="10-Q", accn=None):
    d = {"end": end, "val": val, "accn": accn or f"0009999901-{filed[2:4]}-{filed[5:7]}{filed[8:10]}",
         "form": form, "filed": filed}
    if start:
        d["start"] = start
    return d


def companyfacts() -> dict:
    rev, gp, ocf, capex = [], [], [], []
    ytd = {"2025": 0, "2026": 0}
    ocf_q = {"2025": [20, 25, 25, 30], "2026": [30, 30, 30]}
    cap_q = {"2025": [15, 15, 20, 20], "2026": [20, 20, 20]}
    for i, (end, (start, v)) in enumerate(Q.items()):
        y = end[:4]
        filed, form = FILED[end], FORM[end]
        ytd[y] += v
        # Q2 2026 was first reported 5 lower, then restated (below).
        miss = 5 if end == "2026-06-30" else 0
        if form == "10-Q":
            rev.append(_f(start, end, v - miss, filed, form))
            gp.append(_f(start, end, 0.6 * v, filed, form))
        rev.append(_f(f"{y}-01-01", end, ytd[y] - miss, filed, form))
        n = int(end[5:7]) // 3
        ocf.append(_f(f"{y}-01-01", end, sum(ocf_q[y][:n]), filed, form))
        capex.append(_f(f"{y}-01-01", end, sum(cap_q[y][:n]), filed, form))
        if form == "10-K":
            gp.append(_f(f"{y}-01-01", end, 0.6 * ytd[y], filed, form))
            gp.append(_f("2025-10-01", end, 0.6 * v, filed, form))
    # Restatement: Q2 2026 three-month revenue 160 -> 165 in a 10-Q/A.
    rev.append(_f("2026-04-01", "2026-06-30", 165, "2026-08-20", "10-Q/A", "0009999901-26-000999"))
    rev.append(_f("2026-01-01", "2026-06-30", 315, "2026-08-20", "10-Q/A", "0009999901-26-000999"))

    def inst(vals):
        return [_f(None, end, v, FILED[end], FORM[end]) for end, v in vals.items()]

    return {"cik": CIK, "entityName": "Example Compute Corp", "facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            _f(None, "2026-10-31", 1_000.0, "2026-11-05")]}}},
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": rev}},
            "GrossProfit": {"units": {"USD": gp}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ocf}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": capex}},
            "Assets": {"units": {"USD": inst({"2025-12-31": 1800, "2026-06-30": 1900, "2026-09-30": 2000})}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": inst(
                {"2025-12-31": 250, "2026-06-30": 280, "2026-09-30": 300})}},
            "ShortTermInvestments": {"units": {"USD": inst({"2026-09-30": 100})}},
            "LongTermDebt": {"units": {"USD": inst({"2025-12-31": 520, "2026-06-30": 510, "2026-09-30": 500})}},
            "ShortTermBorrowings": {"units": {"USD": inst({"2026-09-30": 50})}},
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": {"units": {"USD": inst({"2025-12-31": 40})}},
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo": {"units": {"USD": inst({"2025-12-31": 60})}},
            "PropertyPlantAndEquipmentNet": {"units": {"USD": inst({"2025-12-31": 700, "2026-06-30": 760, "2026-09-30": 800})}},
            # A concept outside the whitelist, which the fetch script drops.
            "AccountsPayableCurrent": {"units": {"USD": inst({"2026-09-30": 77})}},
        }}}


def submissions(filings: list[dict] | None = None) -> dict:
    f = filings or [{"form": FORM[e], "filingDate": FILED[e], "reportDate": e,
                     "accessionNumber": "x", "items": ""} for e in FILED]
    return {"cik": CIK, "name": "Example Compute Corp", "sic": "3674", "filings": f}


def closes(start: str, days: int, path: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=days)
    return pd.DataFrame({"date": idx.date.astype(str), "close": path[:days], "adj_close": path[:days], "volume": 0})


def write_pull(root: Path, who: str = "universe", cid: str = "example", facts: dict | None = None,
               subs: dict | None = None, prices: pd.DataFrame | None = None, cik: int = CIK) -> Path:
    (root / "facts").mkdir(parents=True, exist_ok=True)
    (root / "submissions").mkdir(parents=True, exist_ok=True)
    (root / "facts" / f"{cik}.json").write_text(json.dumps(facts or companyfacts()))
    (root / "submissions" / f"{cik}.json").write_text(json.dumps(subs or submissions()))
    cmap = root / "cik_map.csv"
    row = pd.DataFrame([{"who": who, "id": cid, "ticker": "EXC", "name": "Example", "cik": cik}])
    if cmap.exists():
        row = pd.concat([pd.read_csv(cmap), row])
    row.to_csv(cmap, index=False)
    if prices is not None:
        (root / "prices" / who).mkdir(parents=True, exist_ok=True)
        prices.to_csv(root / "prices" / who / f"{cid}.csv", index=False)
    return root
