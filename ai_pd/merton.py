"""Merton-style structural cross-check, with the wider default barrier.

Equity is a call on firm assets V with strike D (debt):

    E(V) = V N(d1) - D e^{-rT} N(d2)

Classic Merton defaults when V_T < D. Under this model's definition a 90%
equity markdown also counts, so the barrier is the higher of D and V*,
where E(V*) = (1 - threshold) * E_ref and E_ref is the last price investors
paid. For a company with no debt, V* = 10% of E_ref.

Simplifications, stated so a reviewer can challenge them:
- V0 is approximated as E_now + D e^{-rT}, not solved from equity.
- sigma_V = sigma_E * E_now / V0.
- Asset drift is set to zero over the horizon.
"""

from __future__ import annotations

import math

import pandas as pd


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def equity_value(v: float, debt: float, sigma_v: float, r: float, t: float) -> float:
    if debt <= 0:
        return v
    if v <= 0:
        return 0.0
    vol = sigma_v * math.sqrt(t)
    d1 = (math.log(v / debt) + (r + 0.5 * sigma_v**2) * t) / vol
    d2 = d1 - vol
    return v * norm_cdf(d1) - debt * math.exp(-r * t) * norm_cdf(d2)


def markdown_barrier(
    equity_ref: float, debt: float, sigma_v: float, r: float, t: float, threshold: float
) -> float:
    """Asset value V* at which equity is worth (1 - threshold) * equity_ref."""
    target = (1.0 - threshold) * equity_ref
    if debt <= 0:
        return target
    lo, hi = 0.0, max(equity_ref + debt, debt) * 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if equity_value(mid, debt, sigma_v, r, t) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def merton_pd(
    equity_now: float,
    equity_ref: float,
    debt: float,
    equity_vol: float,
    *,
    r: float = 0.04,
    t: float = 1.0,
    threshold: float = 0.90,
) -> float:
    """Probability that V_T ends below max(D, V*) within horizon t."""
    if not (equity_ref > 0 and equity_vol > 0 and equity_now == equity_now):
        return float("nan")  # no market data (NaN fails every comparison)
    if equity_now <= (1.0 - threshold) * equity_ref:
        return 1.0  # already marked down past the threshold
    v0 = equity_now + max(debt, 0.0) * math.exp(-r * t)
    sigma_v = equity_vol * equity_now / v0
    barrier = max(debt, markdown_barrier(equity_ref, debt, sigma_v, r, t, threshold))
    if barrier <= 0:
        return 0.0
    dd = (math.log(v0 / barrier) - 0.5 * sigma_v**2 * t) / (sigma_v * math.sqrt(t))
    return norm_cdf(-dd)


def merton_pd_frame(df: pd.DataFrame, r: float, threshold: float, t: float = 1.0) -> pd.Series:
    return pd.Series(
        [
            merton_pd(
                row.equity_value_now,
                row.equity_value_ref,
                row.debt,
                row.equity_vol,
                r=r,
                t=t,
                threshold=threshold,
            )
            for row in df.itertuples()
        ],
        index=df.index,
        name="pd_merton_1y",
    )
