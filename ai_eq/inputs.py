"""Read stock inputs from a CSV.

Required columns: name, segment, price, eps_next, gross_margin,
operating_margin. Optional: years_to_first_flow, discount_rate, and per-stock
overrides revenue_elasticity_{low,central,high} and margin_hit_{low,central,high}
(all three or none). Company runs keep their input files outside the repo.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ai_eq.config import CASES
from ai_eq.exposure import Stock

REQUIRED = ["name", "segment", "price", "eps_next", "gross_margin", "operating_margin"]


def _opt(row, col):
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _band(row, stem):
    vals = {c: _opt(row, f"{stem}_{c}") for c in CASES}
    if all(v is None for v in vals.values()):
        return None
    if any(v is None for v in vals.values()):
        raise ValueError(f"{row['name']}: give all of {stem}_low/central/high or none")
    return vals


def read_stocks(path: str | Path) -> list[Stock]:
    return stocks_from_frame(pd.read_csv(path))


def stocks_from_frame(df: pd.DataFrame) -> list[Stock]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = []
    for _, row in df.iterrows():
        row = row.to_dict()
        s = Stock(name=str(row["name"]), segment=str(row["segment"]), price=float(row["price"]),
                  eps_next=float(row["eps_next"]), gross_margin=float(row["gross_margin"]),
                  operating_margin=float(row["operating_margin"]),
                  years_to_first_flow=_opt(row, "years_to_first_flow"),
                  discount_rate=_opt(row, "discount_rate"),
                  revenue_elasticity=_band(row, "revenue_elasticity"),
                  margin_hit=_band(row, "margin_hit"))
        s.check()
        out.append(s)
    return out
