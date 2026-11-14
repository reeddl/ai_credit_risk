"""Turn raw company inputs into feature metrics and 1-5 scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig

# Raw columns every company record must carry.
RAW_COLUMNS = [
    "company_id",
    "segment",
    "public",
    "cash",
    "monthly_net_burn",
    "revenue_annual",
    "revenue_growth",
    "revenue_stage",  # "none", "pilot" or "recurring"
    "gross_margin",
    "margin_path_score",  # 1-5 judgment
    "moat_score",  # 1-5 judgment
    "employees",
    "compute_cost_share",  # compute cost as share of revenue (buyers of compute)
    "compute_revenue_share",  # share of revenue from GPU rental or chip sales (sellers)
    "backlog_cover",  # share of compute revenue under contract
    "gpu_asset_share",  # share of hard assets that are GPUs
    "debt",
    "floating_debt_share",
    "debt_due_24m",
    "total_assets",
    "hard_assets",
    "liquidation_haircut",
    "funding_dependence",  # share of next year's cash need from new money
    "top5_customer_share",
    "ig_customer_share",
    "circular_revenue_share",
    "equity_value_ref",  # last price investors paid, as a total equity value
    "equity_value_now",  # current mark
    "equity_vol",  # annualised equity volatility from comparables
]

NO_LIMIT = 999.0


def check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"company data is missing columns: {missing}")


def _safe_div(num: pd.Series, den: pd.Series, when_zero: float) -> pd.Series:
    den = den.astype(float)
    out = num.astype(float) / den.where(den > 0)
    return out.fillna(when_zero)


def revenue_score(stage: pd.Series, growth: pd.Series) -> pd.Series:
    recurring = np.select(
        [growth >= 0.5, growth >= 0.2, growth >= 0.0], [1, 2, 3], default=4
    )
    score = np.where(stage == "none", 5, np.where(stage == "pilot", 4, recurring))
    return pd.Series(score, index=stage.index, dtype=float)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive every metric the feature table names."""
    check_columns(df)
    m = pd.DataFrame(index=df.index)
    m["compute_exposure"] = df["compute_cost_share"] + df["compute_revenue_share"] * (
        1.0 - df["backlog_cover"]
    )
    m["runway_months"] = _safe_div(df["cash"], df["monthly_net_burn"], NO_LIMIT)
    m["funding_dependence"] = df["funding_dependence"]
    m["revenue_score"] = revenue_score(df["revenue_stage"], df["revenue_growth"])
    # Gross margin means nothing with no revenue: leave it missing (neutral).
    m["gross_margin"] = df["gross_margin"].where(df["revenue_annual"] > 0)
    m["moat_score"] = df["moat_score"]
    m["margin_path_score"] = df["margin_path_score"]
    m["rate_exposure"] = _safe_div(
        df["debt_due_24m"] + df["floating_debt_share"] * df["debt"], df["total_assets"], 0.0
    )
    m["top5_customer_share"] = df["top5_customer_share"].where(df["revenue_annual"] > 0)
    m["debt_to_assets"] = _safe_div(df["debt"], df["total_assets"], 0.0)
    obligations = df["debt"] + 12.0 * df["monthly_net_burn"].clip(lower=0)
    m["collateral_cover"] = _safe_div(
        df["hard_assets"] * (1.0 - df["liquidation_haircut"]), obligations, NO_LIMIT
    )
    m["ig_customer_share"] = df["ig_customer_share"].where(df["revenue_annual"] > 0)
    m["circular_revenue_share"] = df["circular_revenue_share"]
    m["revenue_per_employee"] = _safe_div(df["revenue_annual"], df["employees"], 0.0)
    return m


def bin_score(values: pd.Series, direction: str, bins: list[float] | None) -> pd.Series:
    """Map a metric to a 1 (safest) to 5 (riskiest) score. Missing scores 3."""
    v = values.astype(float)
    if direction == "score":
        out = v.clip(1, 5).round()
    elif direction == "higher_safer":
        out = pd.Series(
            np.select([v >= b for b in bins], [1, 2, 3, 4], default=5), index=v.index, dtype=float
        )
    elif direction == "lower_safer":
        out = pd.Series(
            np.select([v <= b for b in bins], [1, 2, 3, 4], default=5), index=v.index, dtype=float
        )
    else:
        raise ValueError(f"unknown direction {direction}")
    return out.where(v.notna(), 3.0)


def score_features(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Return one 1-5 score column per configured feature."""
    metrics = compute_metrics(df)
    scores = pd.DataFrame(index=df.index)
    for feat_id, feat in cfg.features.items():
        scores[feat_id] = bin_score(metrics[feat["metric"]], feat["direction"], feat.get("bins"))
    return scores
