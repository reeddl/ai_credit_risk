"""Drawdown per stock per scenario, and the headline sensitivity."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_eq.config import CASES, EquityConfig, load_equity_config
from ai_eq.exposure import Stock, eps_elasticity, shocked_eps_factor
from ai_eq.rdcf import implied_growth, value
from ai_eq.revision import k_band, revised_growth
from ai_risk.driver import Driver, load_driver

HEADLINE_CUT = 0.10


@dataclass(frozen=True)
class Valuation:
    r: float
    kw: dict

    @classmethod
    def for_stock(cls, stock: Stock, cfg: EquityConfig) -> "Valuation":
        v = cfg.valuation
        r = v["discount_rate"] if stock.discount_rate is None else stock.discount_rate
        t1 = v["years_to_first_flow"] if stock.years_to_first_flow is None else stock.years_to_first_flow
        return cls(r, dict(years_to_first_flow=t1, explicit_years=int(v["explicit_years"]),
                           fade_years=int(v["fade_years"]), terminal_growth=v["terminal_growth"]))


def drawdown(stock: Stock, cut, case: str = "central", cfg: EquityConfig | None = None,
             driver: Driver | None = None, discount_rate: float | None = None) -> float:
    """Price change from today (negative is a fall) for one stock, cut and case."""
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    cfg = cfg or load_equity_config()
    driver = driver or load_driver()
    val = Valuation.for_stock(stock, cfg)
    r = val.r if discount_rate is None else discount_rate
    c = driver.leg_cut(driver.resolve(cut) if cut is not None else 0.0, cfg.leg(stock.segment))
    g0 = implied_growth(stock.price, stock.eps_next, r, **val.kw)
    e = eps_elasticity(stock, cfg)[case]
    eps = stock.eps_next * shocked_eps_factor(e, c, cfg.valuation["earnings_floor"])
    g = revised_growth(g0, k_band(stock.segment, cfg)[case], c)
    return value(eps, g, r, **val.kw) / stock.price - 1.0


def stock_table(stocks: list[Stock], cuts=None, cfg: EquityConfig | None = None,
                driver: Driver | None = None) -> pd.DataFrame:
    """One row per stock and named cut: low, central and high drawdown."""
    cfg = cfg or load_equity_config()
    driver = driver or load_driver()
    cuts = list(driver.cuts) if cuts is None else list(cuts)
    rows = []
    for s in stocks:
        val = Valuation.for_stock(s, cfg)
        g0 = implied_growth(s.price, s.eps_next, val.r, **val.kw)
        el = eps_elasticity(s, cfg)
        for cut in cuts:
            c = driver.resolve(cut)
            row = {"name": s.name, "segment": s.segment, "scenario": cut if isinstance(cut, str) else f"{c:.0%}",
                   "cut": c, "leg_cut": driver.leg_cut(c, cfg.leg(s.segment)),
                   "implied_growth": g0, "eps_elasticity_central": el["central"]}
            for case in CASES:
                row[f"drawdown_{case}"] = drawdown(s, c, case, cfg, driver)
            rows.append(row)
    return pd.DataFrame(rows)


def headline(stocks: list[Stock], cfg: EquityConfig | None = None,
             driver: Driver | None = None) -> pd.DataFrame:
    """Central % price move per 10% cut, most exposed first."""
    cfg = cfg or load_equity_config()
    driver = driver or load_driver()
    rows = [{"name": s.name, "segment": s.segment,
             "move_per_10pct_cut": drawdown(s, HEADLINE_CUT, "central", cfg, driver)}
            for s in stocks]
    return pd.DataFrame(rows).sort_values("move_per_10pct_cut").reset_index(drop=True)
