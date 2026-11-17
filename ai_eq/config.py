"""Load and check config/equity.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_risk.driver import LEGS

DEFAULT_EQUITY = Path(__file__).resolve().parent.parent / "config" / "equity.toml"
CASES = ("low", "central", "high")


@dataclass(frozen=True)
class EquityConfig:
    raw: dict[str, Any]

    @property
    def valuation(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["valuation"].items()}

    @property
    def segments(self) -> dict[str, dict[str, Any]]:
        return self.raw["segments"]

    def segment(self, name: str) -> dict[str, Any]:
        if name not in self.segments:
            raise ValueError(f"no equity settings for segment {name!r}; have {sorted(self.segments)}")
        return self.segments[name]

    def _banded(self, table: str, segment: str) -> dict[str, float]:
        node = self.raw[table]
        band = node.get("segments", {}).get(segment, node["default"])
        return {c: float(band[c]) for c in CASES}

    def kappa(self, segment: str) -> dict[str, float]:
        return self._banded("revision", segment)

    @property
    def floor_pe(self) -> float:
        return float(self.raw["revision"]["floor_pe"])

    def margin_hit(self, segment: str) -> dict[str, float]:
        return self._banded("margin_hit", segment)

    def revenue_elasticity(self, segment: str) -> dict[str, float]:
        lo, mid, hi = self.segment(segment)["revenue_elasticity"]
        return {"low": float(lo), "central": float(mid), "high": float(hi)}

    def leg(self, segment: str) -> str:
        return self.segment(segment)["driver_leg"]


def _ordered(band: dict[str, float], what: str) -> None:
    if not band["low"] <= band["central"] <= band["high"]:
        raise ValueError(f"{what}: need low <= central <= high, got {band}")


def validate(cfg: EquityConfig) -> None:
    v = cfg.valuation
    if not v["discount_rate"] > v["terminal_growth"]:
        raise ValueError("discount_rate must exceed terminal_growth")
    if v["explicit_years"] < 0 or v["fade_years"] < 1:
        raise ValueError("explicit_years must be >= 0 and fade_years >= 1")
    if not 0.0 < v["earnings_floor"] < 1.0:
        raise ValueError("earnings_floor must be in (0, 1)")
    if not 1.0 < cfg.floor_pe < 1.0 / (v["discount_rate"] - v["terminal_growth"]):
        raise ValueError("floor_pe must be above 1 and below the no-growth multiple")
    if v["years_to_first_flow"] < 0:
        raise ValueError("years_to_first_flow must be >= 0")
    for name in cfg.segments:
        if cfg.leg(name) not in LEGS:
            raise ValueError(f"segment {name}: unknown driver_leg {cfg.leg(name)!r}")
        re = cfg.revenue_elasticity(name)
        _ordered(re, f"segment {name} revenue_elasticity")
        if re["low"] < 0:
            raise ValueError(f"segment {name}: revenue_elasticity must be >= 0")
        _ordered(cfg.kappa(name), f"revision kappa for {name}")
        _ordered(cfg.margin_hit(name), f"margin_hit for {name}")
        if cfg.kappa(name)["low"] < 0 or cfg.margin_hit(name)["low"] < 0:
            raise ValueError(f"segment {name}: kappa and margin_hit must be >= 0")


def load_equity_config(path: str | Path = DEFAULT_EQUITY) -> EquityConfig:
    with open(path, "rb") as fh:
        cfg = EquityConfig(tomllib.load(fh))
    validate(cfg)
    return cfg
