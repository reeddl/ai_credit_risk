"""One-at-a-time sensitivity: which assumptions move PDs most."""

from __future__ import annotations

import pandas as pd

from .config import ModelConfig
from .pipeline import mean_pd

WEIGHT_SHIFTS = (0.75, 1.25)
PRIOR_SHIFTS = (0.5, 2.0)


def sensitivity_table(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Mean base 1-year PD across the universe, with each assumption shifted.

    Feature weights move by -25% / +25%; segment priors are halved and
    doubled. Rows are sorted by the size of the swing.
    """
    base = mean_pd(df, cfg)
    rows = []

    for feat_id in cfg.features:
        lo_hi = []
        for factor in WEIGHT_SHIFTS:
            c = cfg.copy()
            c.features[feat_id]["weight"] *= factor
            lo_hi.append(mean_pd(df, c))
        rows.append(("feature weight", feat_id, *lo_hi))

    for seg_id in cfg.segments:
        if seg_id not in set(df["segment"]):
            continue
        lo_hi = []
        for factor in PRIOR_SHIFTS:
            c = cfg.copy()
            c.segments[seg_id]["prior_pd_1y"] = min(c.segments[seg_id]["prior_pd_1y"] * factor, 0.99)
            lo_hi.append(mean_pd(df, c))
        rows.append(("segment prior", seg_id, *lo_hi))

    table = pd.DataFrame(rows, columns=["assumption", "item", "mean_pd_low", "mean_pd_high"])
    table["base_mean_pd"] = base
    table["swing"] = (table["mean_pd_high"] - table["mean_pd_low"]).abs()
    return table.sort_values("swing", ascending=False, ignore_index=True)
