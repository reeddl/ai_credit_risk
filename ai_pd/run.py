"""Command line entry point.

    python -m ai_pd.run --n-per-segment 5 --seed 7 --out output
    python -m ai_pd.run --companies my_companies.csv --out output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .pipeline import run_model
from .report import write_report
from .sensitivity import sensitivity_table
from .synthetic import generate_universe


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the AI company default model.")
    ap.add_argument("--config", type=Path, default=None, help="model TOML (default: config/model.toml)")
    ap.add_argument("--companies", type=Path, default=None, help="CSV of company inputs; synthetic if omitted")
    ap.add_argument("--n-per-segment", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.companies:
        df = pd.read_csv(args.companies)
    else:
        df = generate_universe(cfg, args.n_per_segment, args.seed)

    results = run_model(df, cfg)
    sens = sensitivity_table(df, cfg)
    write_report(results, sens, cfg, args.out)
    print(f"Wrote results for {len(df)} companies to {args.out}/")


if __name__ == "__main__":
    main()
