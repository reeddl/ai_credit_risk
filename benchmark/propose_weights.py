"""Fit a feature-weight scale for chosen segments against CDS-implied PDs.

    python benchmark/propose_weights.py [--segments chip_designer fab_equipment]

The benchmark showed the model's PDs for investment-grade chip and fab names
sitting below even a rough physical credit floor: their many score-1
features each subtract log-odds, so they end up far below their prior. This
script asks how much the segment's feature effect should shrink so that each
traded name in those segments lands inside the band

    rough physical CDS PD  <=  model PD  <=  risk-neutral CDS PD

The lower edge is the floor (the model's default event is wider than a
credit event, so it should not be lower). The upper edge is a ceiling, since
the risk-neutral PD includes a risk premium. The scale is fitted by least
squares to the middle of the band in log-odds, with segment priors held
fixed. It also reports the alternative of moving priors instead.

The fit rests on six names and on the placeholder risk-neutral to physical
ratios in ai_pd/market_implied.py; read it as a proposal, not a calibration.

Writes benchmark/results/weight_proposal.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_pd.benchmark import SNAPSHOTS, _md, _pct, compare, load_data, rank_stats  # noqa: E402
from ai_pd.config import ModelConfig, load_config  # noqa: E402
from ai_pd.scorecard import inv_logit, logit  # noqa: E402

DEFAULT_SEGMENTS = ("chip_designer", "fab_equipment")
GRID = np.round(np.arange(0.20, 1.0001, 0.05), 2)


def with_scale(cfg: ModelConfig, segments, scale: float) -> ModelConfig:
    c = cfg.copy()
    for s in segments:
        c.segments[s]["weight_scale"] = float(scale)
    return c


def band(comp: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return logit(comp["pd_physical_1y_est"]), logit(comp["pd_rn_1y"])


def fit_scale(comp_unit: pd.DataFrame, cfg: ModelConfig) -> float:
    """Least-squares scale to the band midpoint, given a run at scale 1."""
    lo, hi = band(comp_unit)
    mid = (lo + hi) / 2
    prior = logit(comp_unit["segment"].map(lambda s: cfg.segments[s]["prior_pd_1y"]))
    feat = logit(comp_unit["pd_1y"]) - prior  # unscaled feature log-odds
    return float(((mid - prior) * feat).sum() / (feat**2).sum())


def in_band(comp: pd.DataFrame) -> pd.Series:
    lo, hi = band(comp)
    x = logit(comp["pd_1y"])
    return (x >= lo - 1e-9) & (x <= hi + 1e-9)


def spread(comp: pd.DataFrame, segment: str, col: str = "pd_1y") -> float:
    x = comp.loc[comp["segment"] == segment, col]
    return float(x.max() / x.min())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segments", nargs="+", default=list(DEFAULT_SEGMENTS))
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark" / "results" / "weight_proposal.md")
    args = ap.parse_args(argv)
    segs = args.segments
    data = load_data()
    base_cfg = with_scale(load_config(), segs, 1.0)

    unit = compare(data, base_cfg, "latest")
    tgt = unit[unit["segment"].isin(segs)].copy()
    lam = fit_scale(tgt, base_cfg)
    lam_r = round(lam / 0.05) * 0.05
    new_cfg = with_scale(base_cfg, segs, lam_r)

    feasible = [g for g in GRID if in_band(compare(data, with_scale(base_cfg, segs, g), "latest").query("segment in @segs")).all()]
    per_seg = {s: fit_scale(tgt[tgt["segment"] == s], base_cfg) for s in segs}

    # Alternative: leave weights alone and move each prior to the band midpoint.
    alt_cfg = base_cfg.copy()
    lo, hi = band(tgt)
    tgt["feat"] = logit(tgt["pd_1y"]) - logit(tgt["segment"].map(lambda s: base_cfg.segments[s]["prior_pd_1y"]))
    tgt["mid"] = (lo + hi) / 2
    alt_priors = {s: float(inv_logit((g["mid"] - g["feat"]).mean())) for s, g in tgt.groupby("segment")}
    for s, p in alt_priors.items():
        alt_cfg.segments[s]["prior_pd_1y"] = p
    alt = compare(data, alt_cfg, "latest").set_index("company_id")

    new = compare(data, new_cfg, "latest").set_index("company_id")
    old = unit.set_index("company_id")
    rows = old[old["segment"].isin(segs)].index
    names = pd.DataFrame(
        {
            "Company": old.loc[rows, "name"],
            "5y CDS, bps": old.loc[rows, "spread_bps"].map(lambda x: f"{x:g}"),
            "Floor (physical est.)": old.loc[rows, "pd_physical_1y_est"].map(_pct),
            "Ceiling (risk-neutral)": old.loc[rows, "pd_rn_1y"].map(_pct),
            "Model PD now": old.loc[rows, "pd_1y"].map(_pct),
            f"Model PD, scale {lam_r:.2f}": new.loc[rows, "pd_1y"].map(_pct),
            "In band": in_band(new.loc[rows]).map({True: "yes", False: "no"}),
            "Priors-only alternative": alt.loc[rows, "pd_1y"].map(_pct),
        }
    )

    weights = pd.DataFrame(
        {
            "Feature": list(base_cfg.features),
            "Weight now": [f"{f['weight']:.3f}" for f in base_cfg.features.values()],
            "Proposed for chip and fab": [f"{f['weight'] * lam_r:.3f}" for f in base_cfg.features.values()],
        }
    )

    def stats(cfg, label):
        out = []
        for snap in SNAPSHOTS:
            comp = compare(data, cfg, snap)
            a, f = rank_stats(comp, "poor"), rank_stats(comp, "approximate")
            out.append(
                f"| {label} | {snap} | {a['spearman_model_vs_cds']:.2f} (p {a['perm_pvalue']:.3f}) | "
                f"{f['spearman_model_vs_cds']:.2f} (p {f['perm_pvalue']:.3f}) | {a['logodds_slope']:.2f} | "
                f"{a['median_model_over_rn']:.2f}x |"
            )
        return out

    lines = [
        "# Proposal: feature-weight scale for chip and fab segments",
        "",
        "Generated by `python benchmark/propose_weights.py`. Nothing here is fitted to defaults;",
        "it is fitted to six CDS quotes and placeholder risk-premium ratios.",
        "",
        "## Proposal",
        "",
        f"Multiply every feature weight by **{lam_r:.2f}** for {', '.join(segs)} "
        f"(least-squares fit {lam:.3f}, rounded), and leave the segment priors where they are. "
        f"Other segments keep their weights. In config this is `weight_scale = {lam_r:.2f}` on each segment.",
        "",
        f"Any scale from {min(feasible):.2f} to {max(feasible):.2f} puts all six names inside the band; "
        "fitted separately, the segments give "
        + ", ".join(f"{s} {v:.2f}" for s, v in per_seg.items())
        + ".",
        "",
        "## Weights",
        "",
        _md(weights),
        "",
        "## PDs against CDS (latest quotes)",
        "",
        "The band runs from the rough physical credit PD (floor: the model's default event is wider",
        "than a credit event) to the risk-neutral CDS PD (ceiling: it includes a risk premium).",
        "",
        _md(names),
        "",
        "## Ranking across all 12 names",
        "",
        "| Version | Snapshot | Spearman, all 12 | Spearman, 9 with fitting segments | Log-odds slope | Median model / RN |",
        "| --- | --- | --- | --- | --- | --- |",
        *stats(base_cfg, "Now"),
        *stats(new_cfg, f"Scale {lam_r:.2f}"),
        *stats(alt_cfg, "Priors-only alternative"),
        "",
        "## Why not move the priors instead",
        "",
        "Keeping the weights and lifting the priors to the band midpoint needs "
        + ", ".join(f"{s} {_pct(p, 1)}" for s, p in alt_priors.items())
        + " (now "
        + ", ".join(f"{_pct(base_cfg.segments[s]['prior_pd_1y'], 0)}" for s in alt_priors)
        + f"). It puts {int(in_band(alt.loc[rows]).sum())} of {len(rows)} names in the band, but the prior is "
        "the PD of an average-scoring company, so it would rate an average chip designer riskier than an "
        f"average neocloud ({_pct(base_cfg.segments['neocloud']['prior_pd_1y'], 0)}). And it keeps the "
        "current spread between names within each segment. Riskiest to safest name in each segment, "
        "as a multiple: "
        + "; ".join(
            f"{s} CDS {spread(old, s, 'pd_rn_1y'):.1f}x, priors-only {spread(alt, s):.1f}x, "
            f"scale {lam_r:.2f} {spread(new, s):.1f}x"
            for s in segs
        )
        + ".",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"fitted scale {lam:.3f} -> {lam_r:.2f}; feasible {min(feasible):.2f}-{max(feasible):.2f}; wrote {args.out}")


if __name__ == "__main__":
    main()
