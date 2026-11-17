"""Run the model end to end on a set of companies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import implied_grade
from ai_risk.driver import Driver

from .config import ModelConfig
from .features import score_features
from .merton import merton_pd_frame
from .scenarios import apply_scenario, severities
from .scorecard import logit, score_pd


@dataclass
class Results:
    companies: pd.DataFrame  # one row per company: base, Merton, stressed PDs
    scores: pd.DataFrame  # base feature scores
    stressed_scores: dict[str, pd.DataFrame]


def run_base(df: pd.DataFrame, cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = score_features(df, cfg)
    pds = score_pd(df["segment"], scores, cfg)
    return scores, pds


def run_model(
    df: pd.DataFrame,
    cfg: ModelConfig,
    cut: float | str | None = None,
    driver: Driver | None = None,
) -> Results:
    """Base, Merton and stressed PDs.

    ``cut`` is the cut to the compute demand estimate that the driver-linked
    scenarios run at: a fraction, a named cut (mild, stall, bust) or None
    for the reference cut, which reproduces version 1.
    """
    threshold = cfg.default_definition["markdown_threshold"]
    r = cfg.model["risk_free_rate"]
    gap_flag = cfg.model["merton_gap_flag_logodds"]

    scores, base = run_base(df, cfg)
    out = df[["company_id", "segment", "public"]].copy()
    out["pd_1y"] = base["pd_1y"]
    out["pd_3y"] = base["pd_3y"]
    out["implied_grade"] = implied_grade(base["pd_1y"], cfg)
    out["logodds_prior"] = base["logodds_prior"]
    out["logodds_features"] = base["logodds_features"]

    out["pd_merton_1y"] = merton_pd_frame(df, r=r, threshold=threshold)
    gap = (logit(out["pd_1y"]) - logit(out["pd_merton_1y"])).abs()
    # Big gaps get flagged for review, never averaged away.
    out["merton_gap_flag"] = (gap > gap_flag) & out["pd_merton_1y"].notna()
    out["already_marked_down"] = df["equity_value_now"] <= (1 - threshold) * df["equity_value_ref"]

    stressed_scores: dict[str, pd.DataFrame] = {}
    for name in cfg.scenarios:
        shocked = apply_scenario(df, name, cfg, cut, driver)
        s_scores = score_features(shocked, cfg)
        s_pd = score_pd(df["segment"], s_scores, cfg, severities(name, cfg, cut, driver))
        out[f"pd_1y_{name}"] = s_pd["pd_1y"]
        out[f"pd_merton_1y_{name}"] = merton_pd_frame(shocked, r=r, threshold=threshold)
        stressed_scores[name] = s_scores

    return Results(companies=out, scores=scores, stressed_scores=stressed_scores)


def segment_summary(results: Results, cfg: ModelConfig) -> pd.DataFrame:
    cols = ["pd_1y", "pd_3y", "pd_merton_1y"] + [f"pd_1y_{n}" for n in cfg.scenarios]
    summary = results.companies.groupby("segment")[cols].mean()
    summary.insert(0, "segment_name", [cfg.segments[s]["name"] for s in summary.index])
    summary.insert(1, "prior_pd_1y", [cfg.segments[s]["prior_pd_1y"] for s in summary.index])
    summary["merton_flags"] = results.companies.groupby("segment")["merton_gap_flag"].sum()
    order = [s for s in cfg.segments if s in summary.index]
    return summary.loc[order]


def mean_pd(df: pd.DataFrame, cfg: ModelConfig) -> float:
    _, pds = run_base(df, cfg)
    return float(np.mean(pds["pd_1y"]))
