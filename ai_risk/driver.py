"""The compute demand driver.

One estimate, next calendar year's AI capex by the big four hyperscalers,
drives every stress scenario. A scenario is a fractional cut ``c`` to that
estimate. A second, credit-funded leg (neoclouds, labs, sovereigns) is cut
harder: ``min(beta * c, 1)``.

The credit model's scenarios name a leg and keep the shock sizes written in
``config/model.toml``; those sizes apply at the reference cut and scale in
proportion to the leg's cut elsewhere (see ``shock_scale``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DRIVER = Path(__file__).resolve().parent.parent / "config" / "driver.toml"

LEGS = ("hyperscaler", "credit_funded")


@dataclass(frozen=True)
class Driver:
    raw: dict[str, Any]

    @property
    def consensus(self) -> float:
        return float(self.raw["driver"]["consensus"])

    @property
    def prior_year(self) -> float:
        return float(self.raw["driver"]["prior_year"])

    @property
    def leg2_beta(self) -> float:
        return float(self.raw["driver"]["leg2"]["beta"])

    @property
    def cuts(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["driver"]["cuts"].items()}

    @property
    def reference_cut(self) -> float:
        return float(self.raw["driver"]["credit_reference"]["cut"])

    def resolve(self, cut: float | str | None) -> float:
        """A cut given as a number, a named cut, or None for the reference."""
        if cut is None:
            return self.reference_cut
        if isinstance(cut, str):
            if cut not in self.cuts:
                raise ValueError(f"unknown named cut {cut!r}; have {sorted(self.cuts)}")
            return self.cuts[cut]
        c = float(cut)
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"cut must be in [0, 1], got {c}")
        return c

    def leg_cut(self, cut: float | str | None, leg: str) -> float:
        c = self.resolve(cut)
        if leg == "hyperscaler":
            return c
        if leg == "credit_funded":
            return min(self.leg2_beta * c, 1.0)
        raise ValueError(f"unknown leg {leg!r}; have {LEGS}")

    def level(self, cut: float | str | None) -> float:
        """Leg 1 capex after the cut, in the driver's unit."""
        return self.consensus * (1.0 - self.resolve(cut))

    def growth(self, cut: float | str | None) -> float:
        """Leg 1 year-on-year growth after the cut."""
        return self.level(cut) / self.prior_year - 1.0

    def shock_scale(self, cut: float | str | None, leg: str) -> float:
        """Multiplier on a scenario's written shock sizes for this cut."""
        ref = self.leg_cut(self.reference_cut, leg)
        return self.leg_cut(cut, leg) / ref


def validate_driver(d: Driver) -> None:
    if d.consensus <= 0 or d.prior_year <= 0:
        raise ValueError("driver consensus and prior_year must be positive")
    if d.leg2_beta < 1.0:
        raise ValueError(f"leg2 beta must be at least 1 (the credit-funded leg cuts harder), got {d.leg2_beta}")
    for name, c in d.cuts.items():
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"named cut {name} must be in [0, 1], got {c}")
    if not 0.0 < d.reference_cut <= 1.0:
        raise ValueError(f"reference cut must be in (0, 1], got {d.reference_cut}")


def load_driver(path: str | Path | None = None) -> Driver:
    with open(path or DEFAULT_DRIVER, "rb") as fh:
        d = Driver(tomllib.load(fh))
    validate_driver(d)
    return d
