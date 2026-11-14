"""Default probabilities implied by credit spreads.

Credit triangle (flat hazard rate, constant recovery):

    lambda = s / (1 - R)
    PD(T)  = 1 - exp(-lambda * T)

s is the CDS spread (or a bond's spread over Treasuries) as a decimal per
year, R is the recovery rate. This is the standard first-order approximation
to a bootstrapped ISDA curve. It is close for spreads under about 1,000 bps
and for a flat curve; it overstates near-term PD when the curve slopes up
(most investment-grade names) and understates it when the curve is inverted
(distressed names, where 1-year protection costs more than 5-year).

What these numbers are and are not:

- They are risk-neutral. Spreads pay for expected loss plus a risk premium,
  a liquidity premium and, for CDS, counterparty and supply/demand effects
  (in 2026 heavy hedging of AI capex debt pushed several names wider).
  Physical (real-world) PDs are lower, by a large factor for investment
  grade and a small one for distressed names.
- They price credit events only: bankruptcy, failure to pay and, for some
  contracts, restructuring. This model's default definition also counts a
  shutdown and a 90% equity markdown, so for a company with traded credit
  the model's PD should sit at or above a physical credit PD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_RECOVERY = 0.40  # ISDA standard for senior unsecured North American corporates

# Ratio of risk-neutral to physical default intensity, by spread band.
# ILLUSTRATIVE PLACEHOLDERS that follow the published pattern (the ratio is
# large for high-quality credits and falls toward 1-2 for distressed ones;
# see Hull, Predescu and White 2005, "Bond prices, default probabilities and
# risk premiums"). Replace with a sourced table before relying on the
# physical column. Each entry: (upper spread bound in bps, ratio).
RN_TO_PHYSICAL_RATIO: tuple[tuple[float, float], ...] = (
    (75.0, 8.0),
    (150.0, 5.0),
    (300.0, 3.0),
    (600.0, 2.0),
    (math.inf, 1.5),
)


def hazard_from_spread(spread_bps: float, recovery: float = DEFAULT_RECOVERY) -> float:
    """Flat default intensity (per year) from a spread in basis points."""
    if not 0.0 <= recovery < 1.0:
        raise ValueError(f"recovery must be in [0, 1), got {recovery}")
    if spread_bps < 0:
        raise ValueError(f"spread cannot be negative, got {spread_bps}")
    return (spread_bps / 10_000.0) / (1.0 - recovery)


def pd_from_hazard(hazard: float, horizon_years: float = 1.0) -> float:
    if hazard < 0 or horizon_years < 0:
        raise ValueError("hazard and horizon must be non-negative")
    return 1.0 - math.exp(-hazard * horizon_years)


def implied_pd(spread_bps: float, horizon_years: float = 1.0, recovery: float = DEFAULT_RECOVERY) -> float:
    """Risk-neutral cumulative PD over the horizon."""
    return pd_from_hazard(hazard_from_spread(spread_bps, recovery), horizon_years)


def spread_from_pd(pd_: float, horizon_years: float = 1.0, recovery: float = DEFAULT_RECOVERY) -> float:
    """Inverse of implied_pd, in basis points."""
    if not 0.0 <= pd_ < 1.0:
        raise ValueError(f"pd must be in [0, 1), got {pd_}")
    hazard = -math.log(1.0 - pd_) / horizon_years
    return hazard * (1.0 - recovery) * 10_000.0


def rn_to_physical_ratio(spread_bps: float) -> float:
    for upper, ratio in RN_TO_PHYSICAL_RATIO:
        if spread_bps <= upper:
            return ratio
    return RN_TO_PHYSICAL_RATIO[-1][1]


def physical_pd_estimate(spread_bps: float, horizon_years: float = 1.0, recovery: float = DEFAULT_RECOVERY) -> float:
    """Rough real-world PD: risk-neutral hazard divided by a band ratio."""
    hazard = hazard_from_spread(spread_bps, recovery) / rn_to_physical_ratio(spread_bps)
    return pd_from_hazard(hazard, horizon_years)


@dataclass(frozen=True)
class SpreadQuote:
    company_id: str
    instrument: str  # "cds" or "bond_spread"
    tenor_years: float
    spread_bps: float
    as_of: str
    source: str


def implied_pd_frame(quotes: pd.DataFrame, recovery: float = DEFAULT_RECOVERY) -> pd.DataFrame:
    """Add risk-neutral 1y/5y PDs and a rough physical 1y PD to a quote table."""
    out = quotes.copy()
    s = out["spread_bps"].astype(float)
    out["hazard_rn"] = [hazard_from_spread(x, recovery) for x in s]
    out["pd_rn_1y"] = [implied_pd(x, 1.0, recovery) for x in s]
    out["pd_rn_5y"] = [implied_pd(x, 5.0, recovery) for x in s]
    out["rn_to_physical_ratio"] = [rn_to_physical_ratio(x) for x in s]
    out["pd_physical_1y_est"] = [physical_pd_estimate(x, 1.0, recovery) for x in s]
    out["recovery"] = recovery
    return out


# ---------------------------------------------------------------------------
# Rank statistics, implemented here so the package needs only numpy/pandas.
# ---------------------------------------------------------------------------


def spearman(x, y) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    a = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    b = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def kendall_tau_b(x, y) -> float:
    """Kendall's tau-b, O(n^2), fine for benchmark-sized samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float("nan") if denom == 0 else (concordant - discordant) / denom


def spearman_permutation_pvalue(x, y, n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided permutation p-value for Spearman's rho."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = abs(spearman(x, y))
    if math.isnan(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(ry)
        if abs(np.corrcoef(rx, perm)[0, 1]) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)
