"""Segment prior plus feature scores, combined on the log-odds scale.

    logit(PD_i) = logit(PD_seg(i)) + sum_k w_k * z_ik + gamma_seg(i) * s

z_ik = score_ik - 3, so a company scoring 3 on every feature keeps its
segment prior. s is the scenario severity (0 in the base case).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SCENARIO_DRIVERS, ModelConfig


def logit(p: np.ndarray | pd.Series | float):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def pd_3y_from_1y(pd_1y):
    """Constant-hazard assumption: PD_3y = 1 - (1 - PD_1y)^3."""
    return 1.0 - (1.0 - pd_1y) ** 3


def segment_prior_logodds(segments: pd.Series, cfg: ModelConfig) -> pd.Series:
    unknown = set(segments) - set(cfg.segments)
    if unknown:
        raise ValueError(f"unknown segments: {sorted(unknown)}")
    priors = segments.map(lambda s: cfg.segments[s]["prior_pd_1y"]).astype(float)
    return pd.Series(logit(priors), index=segments.index)


def feature_contribution(scores: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Log-odds contribution of each feature (w_k * z_ik)."""
    contrib = pd.DataFrame(index=scores.index)
    for feat_id, feat in cfg.features.items():
        contrib[feat_id] = feat["weight"] * (scores[feat_id] - 3.0)
    return contrib


def scenario_term(segments: pd.Series, severities: dict[str, float], cfg: ModelConfig) -> pd.Series:
    """gamma_seg * s, summed over the drivers a scenario switches on."""
    total = pd.Series(0.0, index=segments.index)
    for driver in SCENARIO_DRIVERS:
        s = severities.get(driver, 0.0)
        if s:
            total += segments.map(lambda seg: cfg.segments[seg][f"gamma_{driver}"]).astype(float) * s
    return total


def score_pd(
    segments: pd.Series,
    scores: pd.DataFrame,
    cfg: ModelConfig,
    severities: dict[str, float] | None = None,
) -> pd.DataFrame:
    """1-year and 3-year PD with the log-odds pieces kept for explanation."""
    prior = segment_prior_logodds(segments, cfg)
    contrib = feature_contribution(scores, cfg)
    shock = scenario_term(segments, severities or {}, cfg)
    total = prior + contrib.sum(axis=1) + shock
    pd_1y = inv_logit(total)
    return pd.DataFrame(
        {
            "logodds_prior": prior,
            "logodds_features": contrib.sum(axis=1),
            "logodds_scenario": shock,
            "pd_1y": pd_1y,
            "pd_3y": pd_3y_from_1y(pd_1y),
        },
        index=segments.index,
    )
