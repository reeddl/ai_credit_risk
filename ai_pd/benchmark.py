"""Benchmark model PDs against PDs implied by credit default swap spreads.

    python -m ai_pd.benchmark --out benchmark/results

Reads three files in benchmark/data/:

- companies_real.csv: model inputs for real companies, built from filings
  (provenance per figure in companies_sources.csv).
- cds_quotes.csv: dated 5-year CDS quotes with their source.
- segment_map.csv: each company's segment and how well the segment fits.

Two snapshots are compared: the latest quote per company (the headline), and
a summer 2026 snapshot (quotes from June to August) as a robustness check,
because the latest quotes are not all from the same week.

Read the results with three caveats in mind. CDS PDs are risk-neutral and
cover credit events only, while the model targets a wider, real-world
event. Only companies with traded credit can be benchmarked, so private
model labs, app makers and most robotics companies are not tested at all.
And a dozen names give a noisy rank correlation.

The model inputs are a single November 2026 snapshot, so the summer
snapshot compares today's model with summer spreads: it tests whether the
ranking survives a different market moment, not a point-in-time model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig, load_config
from .market_implied import (
    DEFAULT_RECOVERY,
    implied_pd_frame,
    kendall_tau_b,
    spearman,
    spearman_permutation_pvalue,
)
from .pipeline import run_model
from .scorecard import logit

DATA_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "data"

SNAPSHOTS = {
    # name: (earliest as_of, latest as_of)
    "latest": ("2026-06-01", "2026-11-14"),
    "summer_2026": ("2026-06-01", "2026-08-31"),
}
FIT_RANK = {"good": 0, "approximate": 1, "poor": 2}


@dataclass
class BenchmarkData:
    companies: pd.DataFrame
    quotes: pd.DataFrame
    segment_map: pd.DataFrame


def load_data(data_dir: Path = DATA_DIR) -> BenchmarkData:
    companies = pd.read_csv(data_dir / "companies_real.csv")
    quotes = pd.read_csv(data_dir / "cds_quotes.csv", parse_dates=["as_of"])
    seg = pd.read_csv(data_dir / "segment_map.csv")
    return BenchmarkData(companies, quotes, seg)


def select_quotes(quotes: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Latest CDS quote per company with start <= as_of <= end."""
    q = quotes[(quotes["instrument"] == "cds") & (quotes["as_of"] >= start) & (quotes["as_of"] <= end)]
    q = q.sort_values(["company_id", "as_of"]).groupby("company_id", as_index=False).tail(1)
    return q.reset_index(drop=True)


def compare(
    data: BenchmarkData, cfg: ModelConfig, snapshot: str = "latest", recovery: float = DEFAULT_RECOVERY
) -> pd.DataFrame:
    """One row per company with a quote in the snapshot: model and market PDs."""
    start, end = SNAPSHOTS[snapshot]
    quotes = implied_pd_frame(select_quotes(data.quotes, start, end), recovery)
    res = run_model(data.companies, cfg).companies
    cols = ["company_id", "segment", "pd_1y", "pd_merton_1y", "pd_1y_combined", "implied_grade"]
    out = res[cols].merge(data.segment_map[["company_id", "name", "fit"]], on="company_id", how="left")
    out = out.merge(
        quotes[["company_id", "spread_bps", "bound", "as_of", "quality", "pd_rn_1y", "pd_rn_5y", "pd_physical_1y_est"]],
        on="company_id",
        how="inner",
    )
    out["logodds_gap_vs_rn"] = logit(out["pd_1y"]) - logit(out["pd_rn_1y"])
    out["logodds_gap_vs_physical"] = logit(out["pd_1y"]) - logit(out["pd_physical_1y_est"])
    out["model_over_rn"] = out["pd_1y"] / out["pd_rn_1y"]
    out["snapshot"] = snapshot
    return out.sort_values("spread_bps", ascending=False, ignore_index=True)


def rank_stats(comp: pd.DataFrame, max_fit: str = "poor") -> dict[str, float]:
    sub = comp[comp["fit"].map(FIT_RANK) <= FIT_RANK[max_fit]]
    n = len(sub)
    return {
        "n": n,
        "spearman_model_vs_cds": spearman(sub["pd_1y"], sub["spread_bps"]),
        "kendall_model_vs_cds": kendall_tau_b(sub["pd_1y"], sub["spread_bps"]),
        "perm_pvalue": spearman_permutation_pvalue(sub["pd_1y"], sub["spread_bps"]) if n >= 4 else float("nan"),
        "spearman_merton_vs_cds": spearman(sub["pd_merton_1y"], sub["spread_bps"]),
        "median_model_over_rn": float(np.median(sub["model_over_rn"])) if n else float("nan"),
        "logodds_slope": logodds_slope(sub["pd_1y"], sub["pd_rn_1y"]),
    }


def logodds_slope(model_pd, market_pd) -> float:
    """OLS slope of model log-odds on market log-odds.

    1 means the model spreads names out as much as the market does; above 1
    means the model's PDs are more dispersed (too low for safe names, too
    high or about right for risky ones). Rank-free, so it complements the
    Spearman correlation.
    """
    x, y = logit(np.asarray(market_pd, float)), logit(np.asarray(model_pd, float))
    if len(x) < 3 or np.var(x) == 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def segment_gaps(comp: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Average log-odds gap per segment, and the prior a credit floor implies.

    implied_prior_floor is the segment prior that would bring the segment's
    mean model log-odds down (or up) to the mean log-odds of the rough
    physical credit PD. Because the model's default event is wider than a
    credit event, read it as a floor for the prior, not a target.
    """
    g = comp.groupby("segment").agg(
        n=("company_id", "size"),
        names=("name", lambda s: ", ".join(s)),
        mean_model_pd=("pd_1y", "mean"),
        mean_rn_pd=("pd_rn_1y", "mean"),
        mean_physical_pd=("pd_physical_1y_est", "mean"),
        gap_vs_rn=("logodds_gap_vs_rn", "mean"),
        gap_vs_physical=("logodds_gap_vs_physical", "mean"),
    )
    g["prior_pd_1y"] = [cfg.segments[s]["prior_pd_1y"] for s in g.index]
    prior_lo = logit(g["prior_pd_1y"])
    g["implied_prior_floor"] = 1.0 / (1.0 + np.exp(-(prior_lo - g["gap_vs_physical"])))
    return g.sort_values("gap_vs_physical", ascending=False)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(x: float, digits: int = 2) -> str:
    return "" if pd.isna(x) else f"{100 * x:.{digits}f}%"


def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _company_table(comp: pd.DataFrame) -> str:
    t = pd.DataFrame(
        {
            "Company": comp["name"],
            "Segment (fit)": comp["segment"] + " (" + comp["fit"] + ")",
            "5y CDS, bps": [
                (">" if b == "lower" else "") + f"{s:g}" for s, b in zip(comp["spread_bps"], comp["bound"])
            ],
            "Quote date": comp["as_of"].dt.strftime("%Y-%m-%d"),
            "CDS PD 1y, risk-neutral": comp["pd_rn_1y"].map(_pct),
            "CDS PD 1y, physical est.": comp["pd_physical_1y_est"].map(_pct),
            "Model PD 1y": comp["pd_1y"].map(_pct),
            "Merton PD 1y": comp["pd_merton_1y"].map(_pct),
            "Model / RN": comp["model_over_rn"].map(lambda x: f"{x:.1f}x"),
        }
    )
    return _md(t)


def _stats_line(label: str, s: dict) -> str:
    return (
        f"| {label} | {s['n']} | {s['spearman_model_vs_cds']:.2f} | {s['kendall_model_vs_cds']:.2f} | "
        f"{s['perm_pvalue']:.3f} | {s['spearman_merton_vs_cds']:.2f} | {s['median_model_over_rn']:.2f}x | "
        f"{s['logodds_slope']:.2f} |"
    )


def write_report(data: BenchmarkData, cfg: ModelConfig, out_dir: Path, recovery: float = DEFAULT_RECOVERY) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CDS benchmark",
        "",
        "Generated by `python -m ai_pd.benchmark`. Model inputs come from public filings",
        "(`benchmark/data/companies_sources.csv`); CDS quotes and their sources are in",
        "`benchmark/data/cds_quotes.csv`. Model parameters are still the version 1 placeholders.",
        "",
        f"CDS PDs use the credit triangle with recovery {recovery:.0%}: hazard = spread / (1 - R),",
        "PD = 1 - exp(-hazard x t). Physical estimates divide the hazard by an illustrative",
        "risk-premium ratio (see `ai_pd/market_implied.py`); treat that column as rough.",
        "",
    ]
    summary: dict = {}
    for snap in SNAPSHOTS:
        comp = compare(data, cfg, snap, recovery)
        comp.to_csv(out_dir / f"comparison_{snap}.csv", index=False)
        stats_all = rank_stats(comp, "poor")
        stats_fit = rank_stats(comp, "approximate")
        summary[snap] = {"all": stats_all, "fit": stats_fit}
        gaps = segment_gaps(comp, cfg)
        gaps.to_csv(out_dir / f"segment_gaps_{snap}.csv")
        start, end = SNAPSHOTS[snap]
        lines += [
            f"## Snapshot: {snap} (quotes {start} to {end}, latest per company)",
            "",
            _company_table(comp),
            "",
            "| Sample | n | Spearman (model vs CDS) | Kendall | Permutation p | Spearman (Merton vs CDS) | Median model / RN PD | Log-odds slope |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            _stats_line("All names", stats_all),
            _stats_line("Good or approximate segment fit", stats_fit),
            "",
            "Log-odds slope: model log-odds regressed on CDS log-odds. 1 = same spread of risk as the market;",
            "above 1 = the model separates safe and risky names more sharply than CDS do.",
            "",
            "Segment gaps (log-odds, model minus CDS). Positive means the model is more pessimistic.",
            "",
            _md(
                pd.DataFrame(
                    {
                        "Segment": gaps.index,
                        "Names": gaps["names"],
                        "Prior PD": gaps["prior_pd_1y"].map(_pct),
                        "Mean model PD": gaps["mean_model_pd"].map(_pct),
                        "Mean CDS PD (RN)": gaps["mean_rn_pd"].map(_pct),
                        "Gap vs RN": gaps["gap_vs_rn"].map(lambda x: f"{x:+.2f}"),
                        "Gap vs physical": gaps["gap_vs_physical"].map(lambda x: f"{x:+.2f}"),
                        "Prior implied by credit floor": gaps["implied_prior_floor"].map(_pct),
                    }
                )
            ),
            "",
        ]
    (out_dir / "benchmark.md").write_text("\n".join(lines))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark model PDs against CDS-implied PDs.")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=Path("benchmark/results"))
    ap.add_argument("--recovery", type=float, default=DEFAULT_RECOVERY)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    summary = write_report(load_data(args.data), cfg, args.out, args.recovery)
    for snap, s in summary.items():
        a = s["all"]
        f = s["fit"]
        print(
            f"{snap}: all n={a['n']} spearman={a['spearman_model_vs_cds']:.2f} (p={a['perm_pvalue']:.3f}); "
            f"fit n={f['n']} spearman={f['spearman_model_vs_cds']:.2f} (p={f['perm_pvalue']:.3f}); "
            f"median model/RN={a['median_model_over_rn']:.2f}x slope={a['logodds_slope']:.2f}"
        )
    print(f"Wrote {args.out}/benchmark.md")


if __name__ == "__main__":
    main()
