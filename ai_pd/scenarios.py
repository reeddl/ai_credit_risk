"""Named stress scenarios: move raw inputs, then re-score.

Each scenario does two things:
1. It shocks a company's raw inputs (prices, revenue, equity value, rates),
   which changes its feature scores and its Merton inputs.
2. Its severity switches on the segment gamma term in the scorecard, for
   segment-wide effects the features don't capture.
"""

from __future__ import annotations

import pandas as pd

from .config import ModelConfig

# Which scorecard gamma each single scenario drives.
DRIVER_OF = {
    "compute_glut": "compute_glut",
    "demand_stall": "demand_stall",
    "funding_freeze": "funding_freeze",
    "rates_up": "rates_up",
}


def _parts(name: str, cfg: ModelConfig) -> list[str]:
    sc = cfg.scenarios[name]
    return list(sc["combine"]) if "combine" in sc else [name]


def severities(name: str, cfg: ModelConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in _parts(name, cfg):
        driver = DRIVER_OF.get(part)
        if driver is None:
            raise ValueError(f"scenario {part} has no scorecard driver")
        out[driver] = out.get(driver, 0.0) + cfg.scenarios[part].get("severity", 1.0)
    return out


def _apply_one(df: pd.DataFrame, sc: dict) -> pd.DataFrame:
    out = df.copy()

    # Compute glut: sellers of compute lose revenue on the uncontracted part;
    # buyers of compute get cheaper inputs; GPU collateral loses value.
    price = sc.get("gpu_price_change", 0.0)
    util = sc.get("utilisation_change", 0.0)
    if price or util:
        exposed = out["compute_revenue_share"] * (1.0 - out["backlog_cover"])
        rev_factor = (1.0 + (price + util) * exposed).clip(lower=0.0)
        lost_revenue = out["revenue_annual"] * (1.0 - rev_factor)
        costs = out["revenue_annual"] * (1.0 - out["gross_margin"])
        new_rev = out["revenue_annual"] * rev_factor
        # Sellers: costs stay, revenue falls, so margin and burn worsen.
        seller_gm = 1.0 - costs / new_rev.where(new_rev > 0)
        out["gross_margin"] = seller_gm.fillna(out["gross_margin"]).clip(lower=-1.0)
        out["monthly_net_burn"] = out["monthly_net_burn"] + lost_revenue / 12.0
        out["revenue_annual"] = new_rev
        # Buyers: compute cost falls with price.
        saving = out["compute_cost_share"] * (-price)
        out["gross_margin"] = (out["gross_margin"] + saving).clip(upper=0.95)
        out["monthly_net_burn"] = out["monthly_net_burn"] - saving * out["revenue_annual"] / 12.0
        out["compute_cost_share"] = out["compute_cost_share"] * (1.0 + price)
        # GPU-backed collateral reprices.
        out["hard_assets"] = out["hard_assets"] * (1.0 + price * out["gpu_asset_share"])
        out["total_assets"] = out["total_assets"] + out["hard_assets"] - df["hard_assets"]

    # Demand stall: revenue falls and growth slows; lost gross profit adds to burn.
    rev_change = sc.get("revenue_change", 0.0)
    if rev_change:
        lost = out["revenue_annual"] * (-rev_change)
        out["monthly_net_burn"] = out["monthly_net_burn"] + lost * out["gross_margin"].clip(lower=0) / 12.0
        out["revenue_annual"] = out["revenue_annual"] * (1.0 + rev_change)
    growth_change = sc.get("growth_change", 0.0)
    if growth_change:
        out["revenue_growth"] = out["revenue_growth"] + growth_change

    # Funding freeze: equity marks fall; companies relying on new money
    # cannot raise, which the funding dependence score already carries.
    eq_change = sc.get("equity_value_change", 0.0)
    if eq_change:
        out["equity_value_now"] = out["equity_value_now"] * (1.0 + eq_change)

    # Rates up: floating debt and debt refinanced in the next 24 months reprice.
    rate = sc.get("rate_change", 0.0)
    if rate:
        repriced = out["floating_debt_share"] * out["debt"] + out["debt_due_24m"]
        out["monthly_net_burn"] = out["monthly_net_burn"] + repriced.clip(upper=out["debt"]) * rate / 12.0

    return out


def apply_scenario(df: pd.DataFrame, name: str, cfg: ModelConfig) -> pd.DataFrame:
    out = df
    for part in _parts(name, cfg):
        out = _apply_one(out, cfg.scenarios[part])
    return out
