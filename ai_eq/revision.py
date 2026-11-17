"""Growth revision coefficient k.

A cut ``c`` to the driver lowers the growth the market prices after next
year by ``k * c``. Until step 3 fits k per segment on analog drawdowns, it
is the pooled placeholder in config/equity.toml.
"""

from __future__ import annotations

from ai_eq.config import EquityConfig


def k_band(segment: str, cfg: EquityConfig) -> dict[str, float]:
    return cfg.k(segment)


def revised_growth(g0: float, k: float, leg_cut: float) -> float:
    return g0 - k * leg_cut
