"""Growth revision, and its fit on analog drawdowns.

When next year's EPS is cut by a fraction ``x``, the growth the market prices
after next year falls by ``kappa * x``, but never below the growth implied by
``floor_pe``, a forward multiple.

Each analog episode gives two dated points for one stock: peak price and the
consensus EPS for a target fiscal year, then a later price and the consensus
EPS for the same year after the cut. The reverse DCF turns each point into an
implied growth, so

    kappa_episode = (g_peak - g_later) / (1 - eps_later / eps_peak)

Episodes with role ``kappa`` set the kappa band (min, median, max). Episodes
with role ``floor`` were already priced near the floor at the peak (a
cyclical at peak earnings) and set ``floor_pe`` instead: their peak forward
P/E. ``excluded``
episodes are kept with their reason.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_eq.config import EquityConfig
from ai_eq.rdcf import implied_growth

DATA = Path(__file__).resolve().parent.parent / "data" / "analogs"
EPISODES = DATA / "episodes.csv"
SOURCES = DATA / "episode_sources.csv"
ROLES = ("kappa", "floor", "excluded")
FIGURES = ("peak_price", "peak_eps", "later_price", "later_eps")
STATUSES = ("sourced", "derived", "approximate")


def kappa_band(segment: str, cfg: EquityConfig) -> dict[str, float]:
    return cfg.kappa(segment)


def growth_cut(g0: float, kappa: float, eps_cut: float, floor: float) -> float:
    """How far implied growth falls; zero for an EPS rise, capped at the floor."""
    return min(kappa * max(eps_cut, 0.0), max(g0 - floor, 0.0))


def revised_growth(g0: float, kappa: float, eps_cut: float, floor: float) -> float:
    return g0 - growth_cut(g0, kappa, eps_cut, floor)


@dataclass(frozen=True)
class EpisodeFit:
    episode: str
    role: str
    g_peak: float
    g_later: float
    eps_cut: float
    price_change: float
    kappa: float


def load_episodes(path: str | Path = EPISODES) -> pd.DataFrame:
    df = pd.read_csv(path)
    bad = set(df["role"]) - set(ROLES)
    if bad:
        raise ValueError(f"unknown roles {bad}")
    return df


def load_sources(path: str | Path = SOURCES) -> pd.DataFrame:
    return pd.read_csv(path)


def fit_episode(row, cfg: EquityConfig, r: float | None = None) -> EpisodeFit:
    v = cfg.valuation
    r = v["discount_rate"] if r is None else r
    kw = dict(years_to_first_flow=float(row["years_to_first_flow"]), explicit_years=int(v["explicit_years"]),
              fade_years=int(v["fade_years"]), terminal_growth=v["terminal_growth"])
    g0 = implied_growth(float(row["peak_price"]), float(row["peak_eps"]), r, **kw)
    g1 = implied_growth(float(row["later_price"]), float(row["later_eps"]), r, **kw)
    cut = 1.0 - float(row["later_eps"]) / float(row["peak_eps"])
    kappa = (g0 - g1) / cut if cut > 0 else float("nan")
    return EpisodeFit(str(row["episode"]), str(row["role"]), g0, g1, cut,
                      float(row["later_price"]) / float(row["peak_price"]) - 1.0, kappa)


def fit(episodes: pd.DataFrame, cfg: EquityConfig, r: float | None = None) -> dict:
    """Per-episode results, the kappa band and the growth floor."""
    used = episodes[episodes["role"] != "excluded"]
    fits = [fit_episode(row, cfg, r) for _, row in used.iterrows()]
    kap = sorted(f.kappa for f in fits if f.role == "kappa")
    if not kap:
        raise ValueError("no kappa episodes to fit")
    band = {"low": kap[0], "central": statistics.median(kap), "high": kap[-1]}
    floor_pes = [float(row["peak_price"]) / float(row["peak_eps"])
                 for _, row in used.iterrows() if row["role"] == "floor"]
    floor_pe = min(floor_pes) if floor_pes else cfg.floor_pe
    return {"episodes": fits, "kappa": band, "floor_pe": floor_pe}
