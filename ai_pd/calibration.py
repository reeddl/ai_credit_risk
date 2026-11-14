"""Map PDs to implied rating grades.

The grade boundaries in config are placeholders. Replace them with a
published agency default study before using implied ratings as an anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig


def implied_grade(pd_1y: pd.Series, cfg: ModelConfig) -> pd.Series:
    grades = cfg.calibration["grades"]
    bounds = np.asarray(cfg.calibration["upper_pd_1y"], dtype=float)
    idx = np.searchsorted(bounds, pd_1y.to_numpy(dtype=float), side="left")
    idx = np.clip(idx, 0, len(grades) - 1)
    return pd.Series([grades[i] for i in idx], index=pd_1y.index, name="implied_grade")
