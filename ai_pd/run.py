"""Command line entry point.

    python -m ai_pd.run --n-per-segment 5 --seed 7 --out output
    python -m ai_pd.run --companies my_companies.csv --out output
    python -m ai_pd.run --compute-cut bust --out output_bust
    python -m ai_pd.run --compute-cut 0.15 --out output_15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_risk.driver import load_driver

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
    ap.add_argument(
        "--compute-cut",
        default=None,
        help="cut to the compute demand estimate for the stress scenarios: a fraction or mild/stall/bust "
        "(default: the driver's reference cut, which reproduces version 1)",
    )
    ap.add_argument("--driver", type=Path, default=None, help="driver TOML (default: config/driver.toml)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.companies:
        df = pd.read_csv(args.companies)
    else:
        df = generate_universe(cfg, args.n_per_segment, args.seed)

    driver = load_driver(args.driver)
    cut = args.compute_cut
    if cut is not None and cut not in driver.cuts:
        cut = float(cut)
    results = run_model(df, cfg, cut, driver)
    sens = sensitivity_table(df, cfg)
    write_report(results, sens, cfg, args.out)
    print(f"Wrote results for {len(df)} companies to {args.out}/")


if __name__ == "__main__":
    main()
