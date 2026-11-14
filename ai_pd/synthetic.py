"""Synthetic AI companies, one profile per segment.

These exist so the whole pipeline runs before any real data goes in. The
profiles are rough, judgment-set shapes of each segment (who burns, who
borrows, who sells compute), not estimates of any real company.
Money is in US dollars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig
from .features import RAW_COLUMNS

M = 1_000_000.0

# Per-segment profile. Money in $m. Shares in [0, 1].
# rev: median annual revenue; growth: mean growth; stage: P(none, pilot, recurring)
# gm: mean gross margin; burn: median monthly net burn; cash_months: median runway
# ccs / crs: compute cost share / compute revenue share; backlog: contracted share
# gpu: GPU share of hard assets; dta: debt to assets; hard: hard asset share of assets
# fund: funding dependence; top5, ig, circ: customer mix; rpe: revenue per employee ($k)
# vol: equity volatility; mult: equity value / revenue; moat, path: mean 1-5 scores (5 = risky)
PROFILES: dict[str, dict] = {
    "closed_model": dict(rev=800, growth=1.2, stage=(0.0, 0.1, 0.9), gm=0.45, burn=150, cash_months=20,
                         ccs=0.45, crs=0.0, backlog=0.0, gpu=0.6, dta=0.10, hard=0.25, fund=0.65,
                         top5=0.35, ig=0.55, circ=0.15, rpe=900, vol=0.90, mult=25, moat=2.5, path=3.0),
    "open_weight_model": dict(rev=40, growth=1.0, stage=(0.2, 0.3, 0.5), gm=0.35, burn=25, cash_months=16,
                              ccs=0.50, crs=0.0, backlog=0.0, gpu=0.6, dta=0.05, hard=0.20, fund=0.75,
                              top5=0.45, ig=0.40, circ=0.10, rpe=300, vol=1.10, mult=40, moat=3.8, path=3.5),
    "app_maker": dict(rev=30, growth=0.8, stage=(0.15, 0.25, 0.6), gm=0.40, burn=4, cash_months=18,
                      ccs=0.35, crs=0.0, backlog=0.0, gpu=0.0, dta=0.05, hard=0.05, fund=0.60,
                      top5=0.30, ig=0.30, circ=0.05, rpe=250, vol=1.10, mult=15, moat=3.8, path=3.3),
    "data_strong_moat": dict(rev=150, growth=0.4, stage=(0.0, 0.05, 0.95), gm=0.70, burn=1, cash_months=36,
                             ccs=0.05, crs=0.0, backlog=0.0, gpu=0.0, dta=0.15, hard=0.10, fund=0.20,
                             top5=0.55, ig=0.60, circ=0.05, rpe=600, vol=0.60, mult=10, moat=1.8, path=2.0),
    "data_weak_moat": dict(rev=40, growth=0.2, stage=(0.0, 0.1, 0.9), gm=0.35, burn=2, cash_months=14,
                           ccs=0.05, crs=0.0, backlog=0.0, gpu=0.0, dta=0.15, hard=0.10, fund=0.45,
                           top5=0.60, ig=0.45, circ=0.05, rpe=120, vol=0.90, mult=4, moat=4.2, path=3.5),
    "neocloud": dict(rev=500, growth=1.0, stage=(0.0, 0.05, 0.95), gm=0.35, burn=20, cash_months=10,
                     ccs=0.0, crs=0.90, backlog=0.55, gpu=0.80, dta=0.55, hard=0.80, fund=0.55,
                     top5=0.70, ig=0.45, circ=0.25, rpe=2500, vol=0.95, mult=6, moat=3.5, path=3.0),
    "power": dict(rev=900, growth=0.15, stage=(0.0, 0.0, 1.0), gm=0.40, burn=-5, cash_months=30,
                  ccs=0.0, crs=0.0, backlog=0.0, gpu=0.0, dta=0.50, hard=0.85, fund=0.30,
                  top5=0.40, ig=0.65, circ=0.02, rpe=1500, vol=0.45, mult=3, moat=2.5, path=2.5),
    "chip_designer": dict(rev=2000, growth=0.5, stage=(0.0, 0.05, 0.95), gm=0.60, burn=-20, cash_months=40,
                          ccs=0.0, crs=0.70, backlog=0.40, gpu=0.0, dta=0.15, hard=0.25, fund=0.15,
                          top5=0.60, ig=0.55, circ=0.10, rpe=1200, vol=0.65, mult=10, moat=2.5, path=2.5),
    "fab_equipment": dict(rev=5000, growth=0.2, stage=(0.0, 0.0, 1.0), gm=0.45, burn=-50, cash_months=40,
                          ccs=0.0, crs=0.40, backlog=0.50, gpu=0.0, dta=0.25, hard=0.60, fund=0.10,
                          top5=0.50, ig=0.70, circ=0.02, rpe=700, vol=0.50, mult=4, moat=2.0, path=2.5),
    "networking": dict(rev=800, growth=0.4, stage=(0.0, 0.0, 1.0), gm=0.45, burn=-5, cash_months=30,
                       ccs=0.0, crs=0.50, backlog=0.35, gpu=0.0, dta=0.25, hard=0.40, fund=0.20,
                       top5=0.55, ig=0.60, circ=0.05, rpe=600, vol=0.65, mult=5, moat=3.0, path=2.8),
    "infra_software": dict(rev=25, growth=0.9, stage=(0.1, 0.2, 0.7), gm=0.60, burn=3, cash_months=20,
                           ccs=0.25, crs=0.0, backlog=0.0, gpu=0.0, dta=0.05, hard=0.05, fund=0.55,
                           top5=0.35, ig=0.45, circ=0.05, rpe=250, vol=1.00, mult=20, moat=3.7, path=3.0),
    "robotics": dict(rev=10, growth=1.5, stage=(0.4, 0.4, 0.2), gm=0.15, burn=8, cash_months=18,
                     ccs=0.15, crs=0.0, backlog=0.0, gpu=0.1, dta=0.10, hard=0.30, fund=0.80,
                     top5=0.60, ig=0.35, circ=0.05, rpe=80, vol=1.20, mult=60, moat=3.3, path=4.0),
    "gpu_lessor": dict(rev=300, growth=0.6, stage=(0.0, 0.0, 1.0), gm=0.55, burn=-2, cash_months=12,
                       ccs=0.0, crs=0.95, backlog=0.70, gpu=0.95, dta=0.70, hard=0.90, fund=0.50,
                       top5=0.65, ig=0.35, circ=0.20, rpe=5000, vol=0.80, mult=3, moat=3.8, path=3.0),
}

PUBLIC_SHARE = {"power": 0.7, "chip_designer": 0.6, "fab_equipment": 0.8, "networking": 0.6}


def _share(rng, mean, n, spread=8.0):
    """Beta draw around a mean share; spread sets how tight it is."""
    mean = float(np.clip(mean, 0.01, 0.99))
    return rng.beta(mean * spread, (1 - mean) * spread, n)


def _lognormal(rng, median, n, sigma=0.5):
    return median * np.exp(rng.normal(0.0, sigma, n))


def _score(rng, mean, n):
    return np.clip(np.round(rng.normal(mean, 0.8, n)), 1, 5)


def generate_segment(segment: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    p = PROFILES[segment]
    stage = rng.choice(["none", "pilot", "recurring"], size=n, p=p["stage"])
    revenue = _lognormal(rng, p["rev"], n, 0.7)
    revenue = np.where(stage == "none", 0.0, np.where(stage == "pilot", revenue * 0.05, revenue))

    # Negative burn means the company generates cash.
    burn = p["burn"] * np.exp(rng.normal(0.0, 0.5, n)) + rng.normal(0.0, abs(p["burn"]) * 0.3, n)
    cash = np.abs(p["burn"]) * _lognormal(rng, p["cash_months"], n, 0.4)

    equity_ref = np.maximum(revenue, p["rev"] * 0.05) * _lognormal(rng, p["mult"], n, 0.5)
    # Most current marks sit near the last price paid; a few are already well down.
    mark = np.clip(rng.normal(1.0, 0.15, n), 0.3, 1.5)
    mark = np.where(rng.random(n) < 0.08, rng.uniform(0.05, 0.4, n), mark)

    dta = _share(rng, p["dta"], n, 6.0) if p["dta"] > 0 else np.zeros(n)
    # Total assets: cash plus hard assets; hard assets set from their share.
    hard_share = _share(rng, p["hard"], n, 6.0)
    hard = cash * hard_share / np.maximum(1 - hard_share, 0.05)
    total_assets = cash + hard
    debt = dta * total_assets
    employees = np.maximum(5, np.round(np.maximum(revenue, p["rev"] * 0.05) * 1000 / _lognormal(rng, p["rpe"], n, 0.4)))

    df = pd.DataFrame(
        {
            "company_id": [f"{segment}_{i + 1:02d}" for i in range(n)],
            "segment": segment,
            "public": rng.random(n) < PUBLIC_SHARE.get(segment, 0.1),
            "cash": cash * M,
            "monthly_net_burn": burn * M,
            "revenue_annual": revenue * M,
            "revenue_growth": rng.normal(p["growth"], 0.4, n),
            "revenue_stage": stage,
            "gross_margin": np.clip(rng.normal(p["gm"], 0.12, n), -0.5, 0.9),
            "margin_path_score": _score(rng, p["path"], n),
            "moat_score": _score(rng, p["moat"], n),
            "employees": employees,
            "compute_cost_share": _share(rng, p["ccs"], n) if p["ccs"] > 0 else np.zeros(n),
            "compute_revenue_share": _share(rng, p["crs"], n, 12.0) if p["crs"] > 0 else np.zeros(n),
            "backlog_cover": _share(rng, p["backlog"], n) if p["backlog"] > 0 else np.zeros(n),
            "gpu_asset_share": _share(rng, p["gpu"], n, 10.0) if p["gpu"] > 0 else np.zeros(n),
            "debt": debt * M,
            "floating_debt_share": _share(rng, 0.5, n, 4.0),
            "debt_due_24m": debt * _share(rng, 0.3, n, 4.0) * M,
            "total_assets": total_assets * M,
            "hard_assets": hard * M,
            "liquidation_haircut": _share(rng, 0.5, n, 6.0),
            "funding_dependence": _share(rng, p["fund"], n, 6.0),
            "top5_customer_share": _share(rng, p["top5"], n, 6.0),
            "ig_customer_share": _share(rng, p["ig"], n, 6.0),
            "circular_revenue_share": _share(rng, p["circ"], n, 10.0),
            "equity_value_ref": equity_ref * M,
            "equity_value_now": equity_ref * mark * M,
            "equity_vol": np.clip(rng.normal(p["vol"], 0.15, n), 0.25, 2.0),
        }
    )
    return df[RAW_COLUMNS]


def generate_universe(cfg: ModelConfig, n_per_segment: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    missing = set(cfg.segments) - set(PROFILES)
    if missing:
        raise ValueError(f"no synthetic profile for segments: {sorted(missing)}")
    frames = [generate_segment(seg, n_per_segment, rng) for seg in cfg.segments]
    return pd.concat(frames, ignore_index=True)
