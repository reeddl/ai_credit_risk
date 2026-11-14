"""Load and validate the model configuration."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "model.toml"

SCENARIO_DRIVERS = ("compute_glut", "demand_stall", "funding_freeze", "rates_up")
DIRECTIONS = ("higher_safer", "lower_safer", "score")


@dataclass
class ModelConfig:
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def default_definition(self) -> dict[str, Any]:
        return self.raw["default_definition"]

    @property
    def segments(self) -> dict[str, dict[str, Any]]:
        return self.raw["segments"]

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.raw["features"]

    @property
    def scenarios(self) -> dict[str, dict[str, Any]]:
        return self.raw["scenarios"]

    @property
    def calibration(self) -> dict[str, Any]:
        return self.raw["calibration"]

    def copy(self) -> "ModelConfig":
        return ModelConfig(copy.deepcopy(self.raw))


def load_config(path: str | Path | None = None) -> ModelConfig:
    with open(path or DEFAULT_CONFIG, "rb") as fh:
        cfg = ModelConfig(tomllib.load(fh))
    validate(cfg)
    return cfg


def validate(cfg: ModelConfig) -> None:
    for seg_id, seg in cfg.segments.items():
        pd = seg["prior_pd_1y"]
        if not 0 < pd < 1:
            raise ValueError(f"segment {seg_id}: prior_pd_1y must be in (0, 1), got {pd}")
        for driver in SCENARIO_DRIVERS:
            seg.setdefault(f"gamma_{driver}", 0.0)

    for feat_id, feat in cfg.features.items():
        if feat["direction"] not in DIRECTIONS:
            raise ValueError(f"feature {feat_id}: unknown direction {feat['direction']}")
        if feat["direction"] != "score":
            bins = feat["bins"]
            if len(bins) != 4:
                raise ValueError(f"feature {feat_id}: needs 4 bin edges, got {len(bins)}")
            ordered = sorted(bins, reverse=feat["direction"] == "higher_safer")
            if list(bins) != ordered:
                raise ValueError(f"feature {feat_id}: bins are out of order for {feat['direction']}")

    for name, sc in cfg.scenarios.items():
        for part in sc.get("combine", []):
            if part not in cfg.scenarios or "combine" in cfg.scenarios[part]:
                raise ValueError(f"scenario {name}: cannot combine {part}")

    cal = cfg.calibration
    if len(cal["grades"]) != len(cal["upper_pd_1y"]):
        raise ValueError("calibration: grades and upper_pd_1y differ in length")
    if list(cal["upper_pd_1y"]) != sorted(cal["upper_pd_1y"]):
        raise ValueError("calibration: upper_pd_1y must increase")
