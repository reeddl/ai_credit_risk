"""Command line: python -m ai_eq.run --stocks my_stocks.csv [--cut bust] [--out dir]"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_eq.config import DEFAULT_EQUITY, load_equity_config
from ai_eq.drawdown import headline, stock_table
from ai_eq.inputs import read_stocks
from ai_risk.driver import DEFAULT_DRIVER, load_driver


def _cut(text: str):
    try:
        return float(text)
    except ValueError:
        return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stocks", required=True, help="CSV of stock inputs (see ai_eq/inputs.py)")
    p.add_argument("--cut", action="append", type=_cut,
                   help="named cut (mild, stall, bust) or a fraction; repeat for several; default all named")
    p.add_argument("--config", default=str(DEFAULT_EQUITY))
    p.add_argument("--driver", default=str(DEFAULT_DRIVER))
    p.add_argument("--out", help="directory for drawdowns.csv and headline.csv")
    a = p.parse_args(argv)
    cfg = load_equity_config(a.config)
    drv = load_driver(a.driver)
    stocks = read_stocks(a.stocks)
    table = stock_table(stocks, a.cut, cfg, drv)
    head = headline(stocks, cfg, drv)
    fmt = {c: "{:+.1%}".format for c in table.columns if c.startswith("drawdown_")}
    print(table.to_string(index=False, formatters=fmt))
    print()
    print(head.to_string(index=False, formatters={"move_per_10pct_cut": "{:+.1%}".format}))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "drawdowns.csv", index=False)
        head.to_csv(out / "headline.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
