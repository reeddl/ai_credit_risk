"""Reverse DCF on next year's earnings.

Cash flow is taken as earnings (fine for asset-light, high-margin names; a
capex-heavy name needs free cash flow instead). The first flow is next
fiscal year's consensus EPS, ``years_to_first_flow`` from today. Earnings
then grow at ``g`` for ``explicit_years``, fade linearly to
``terminal_growth`` over ``fade_years``, and a Gordon terminal value follows.
"""

from __future__ import annotations

PRICE_SEARCH = (-0.6, 0.8)


def value(eps_next: float, g: float, r: float, *, years_to_first_flow: float = 1.0,
          explicit_years: int = 4, fade_years: int = 5, terminal_growth: float = 0.03) -> float:
    if r <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    e = eps_next
    t = years_to_first_flow
    v = e / (1 + r) ** t
    for _ in range(int(explicit_years)):
        e *= 1 + g
        t += 1
        v += e / (1 + r) ** t
    for i in range(1, int(fade_years) + 1):
        e *= 1 + g + (terminal_growth - g) * i / fade_years
        t += 1
        v += e / (1 + r) ** t
    v += e * (1 + terminal_growth) / (r - terminal_growth) / (1 + r) ** t
    return v


def implied_growth(price: float, eps_next: float, r: float, **kw) -> float:
    """Post-first-year growth that makes ``value`` equal ``price``."""
    if price <= 0 or eps_next <= 0:
        raise ValueError("price and next-year EPS must be positive")
    lo, hi = PRICE_SEARCH
    if not value(eps_next, lo, r, **kw) <= price <= value(eps_next, hi, r, **kw):
        raise ValueError(
            f"price {price} is outside what growth in {PRICE_SEARCH} can explain at r={r}")
    for _ in range(200):
        mid = (lo + hi) / 2
        if value(eps_next, mid, r, **kw) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
