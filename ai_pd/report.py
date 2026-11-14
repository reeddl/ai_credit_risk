"""Write results to CSV and a short markdown summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ModelConfig
from .pipeline import Results, segment_summary


def _pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:.1f}%"


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_report(results: Results, sensitivity: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.companies.to_csv(out_dir / "companies.csv", index=False)
    results.scores.assign(company_id=results.companies["company_id"]).to_csv(
        out_dir / "feature_scores.csv", index=False
    )
    sensitivity.to_csv(out_dir / "sensitivity.csv", index=False)

    summary = segment_summary(results, cfg)
    seg = summary.reset_index()
    pct_cols = [c for c in seg.columns if c.startswith("pd_") or c == "prior_pd_1y"]
    for c in pct_cols:
        seg[c] = seg[c].map(_pct)
    seg = seg.drop(columns=["segment"]).rename(columns={"segment_name": "segment"})

    top = sensitivity.head(8).copy()
    for c in ["mean_pd_low", "mean_pd_high", "base_mean_pd", "swing"]:
        top[c] = top[c].map(_pct)

    n = len(results.companies)
    lines = [
        "# AI company default model: version 1 results",
        "",
        f"{n} synthetic companies across {summary.shape[0]} segments. All inputs and",
        "parameters are placeholders; these numbers show the pipeline works, not real risk.",
        "",
        "Default = missed debt payment, bankruptcy filing, shutdown, or a sale or down round",
        f"at least {int(100 * cfg.default_definition['markdown_threshold'])}% below the last price investors paid.",
        "",
        "## Mean PD by segment (1-year unless marked)",
        "",
        _md_table(seg),
        "",
        f"Merton flags: {int(results.companies['merton_gap_flag'].sum())} of {n} companies differ from",
        "the scorecard by more than the configured log-odds gap and need review.",
        "",
        "## Assumptions that move the mean PD most",
        "",
        _md_table(top[["assumption", "item", "mean_pd_low", "mean_pd_high", "swing"]]),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))
