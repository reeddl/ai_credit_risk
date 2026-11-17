"""EPS elasticity to the compute demand driver.

Revenue moves by ``revenue_elasticity`` per unit cut to the driver leg.
Operating income moves by ``(gross_margin + margin_hit) / operating_margin``
per unit of revenue: fixed opex gives the leverage, and ``margin_hit`` is the
gross margin lost per unit of revenue lost as pricing power fades. Earnings
are taken to move with operating income.

The earnings cut is linear in the driver cut: ``1 - e * c``. Operating
leverage is a statement about dollars (each dollar of revenue lost takes
``gross margin + hit`` dollars of profit with it), so earnings fall in a
straight line and reach zero at a finite cut. A constant-elasticity form,
``(1 - c) ** e``, never reaches zero and understates deep cuts. Earnings are
floored at ``earnings_floor`` of consensus so the DCF stays defined.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_eq.config import CASES, EquityConfig


@dataclass(frozen=True)
class Stock:
    name: str
    segment: str
    price: float
    eps_next: float
    gross_margin: float
    operating_margin: float
    years_to_first_flow: float | None = None
    discount_rate: float | None = None
    revenue_elasticity: dict[str, float] | None = None  # per-stock override
    margin_hit: dict[str, float] | None = None          # per-stock override

    def check(self) -> None:
        if self.price <= 0 or self.eps_next <= 0:
            raise ValueError(f"{self.name}: price and eps_next must be positive")
        if not 0 < self.operating_margin <= self.gross_margin <= 1:
            raise ValueError(
                f"{self.name}: need 0 < operating_margin <= gross_margin <= 1; "
                "loss-making names belong to the credit model")


def earnings_leverage(gross_margin: float, operating_margin: float, margin_hit: float) -> float:
    return (gross_margin + margin_hit) / operating_margin


def eps_elasticity(stock: Stock, cfg: EquityConfig) -> dict[str, float]:
    stock.check()
    re = stock.revenue_elasticity or cfg.revenue_elasticity(stock.segment)
    hit = stock.margin_hit or cfg.margin_hit(stock.segment)
    return {c: re[c] * earnings_leverage(stock.gross_margin, stock.operating_margin, hit[c])
            for c in CASES}


def shocked_eps_factor(elasticity: float, leg_cut: float, floor: float) -> float:
    """Shocked EPS as a fraction of consensus."""
    return max(1.0 - elasticity * leg_cut, floor)
