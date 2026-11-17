"""Equity drawdown engine (ai_eq) on fictional example stocks."""

import copy
import io
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ai_eq.config import CASES, DEFAULT_EQUITY, EquityConfig, load_equity_config, validate
from ai_eq.drawdown import HEADLINE_CUT, Valuation, drawdown, headline, stock_table
from ai_eq.exposure import Stock, eps_elasticity, shocked_eps_factor
from ai_eq.inputs import read_stocks
from ai_eq.rdcf import implied_growth, value
from ai_eq.run import main as run_main
from ai_pd.config import load_config
from ai_risk.driver import DEFAULT_DRIVER, Driver, load_driver

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "tests" / "data" / "equity_example_stocks.csv"
CFG = load_equity_config()
DRV = load_driver()
STOCKS = read_stocks(EXAMPLES)
CUTS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]


def cfg_with(fn) -> EquityConfig:
    with open(DEFAULT_EQUITY, "rb") as fh:
        raw = tomllib.load(fh)
    fn(raw)
    return EquityConfig(raw)


def driver_beta(beta: float) -> Driver:
    with open(DEFAULT_DRIVER, "rb") as fh:
        raw = tomllib.load(fh)
    raw["driver"]["leg2"]["beta"] = beta
    return Driver(raw)


class ConfigTest(unittest.TestCase):
    def test_segments_match_credit_model(self):
        self.assertEqual(set(CFG.segments), set(load_config().segments))

    def test_bands_out_of_order_rejected(self):
        bad = cfg_with(lambda r: r["revision"]["default"].update(low=0.6))
        with self.assertRaises(ValueError):
            validate(bad)

    def test_unknown_leg_rejected(self):
        bad = cfg_with(lambda r: r["segments"]["power"].update(driver_leg="sovereign"))
        with self.assertRaises(ValueError):
            validate(bad)

    def test_discount_rate_must_exceed_terminal_growth(self):
        bad = cfg_with(lambda r: r["valuation"].update(discount_rate=0.03))
        with self.assertRaises(ValueError):
            validate(bad)


class ReverseDcfTest(unittest.TestCase):
    def test_implied_growth_reproduces_price(self):
        for s in STOCKS:
            val = Valuation.for_stock(s, CFG)
            for r in (0.09, 0.10, 0.11, 0.12):
                g = implied_growth(s.price, s.eps_next, r, **val.kw)
                self.assertAlmostEqual(value(s.eps_next, g, r, **val.kw), s.price, places=6)

    def test_value_rises_with_growth_and_falls_with_rate(self):
        self.assertLess(value(10, 0.0, 0.10), value(10, 0.05, 0.10))
        self.assertGreater(value(10, 0.0, 0.09), value(10, 0.0, 0.11))

    def test_price_outside_search_range_raises(self):
        with self.assertRaises(ValueError):
            implied_growth(1e6, 1.0, 0.10)

    def test_loss_making_stock_rejected(self):
        with self.assertRaises(ValueError):
            replace(STOCKS[0], operating_margin=-0.1).check()


class ExposureTest(unittest.TestCase):
    def test_elasticity_is_revenue_times_leverage(self):
        s = STOCKS[0]
        re = CFG.revenue_elasticity(s.segment)["central"]
        hit = CFG.margin_hit(s.segment)["central"]
        expect = re * (s.gross_margin + hit) / s.operating_margin
        self.assertAlmostEqual(eps_elasticity(s, CFG)["central"], expect)

    def test_earnings_cut_is_linear_then_floored(self):
        self.assertAlmostEqual(shocked_eps_factor(1.5, 0.2, 0.05), 0.7)
        self.assertAlmostEqual(shocked_eps_factor(1.5, 0.4, 0.05), 0.4)
        self.assertEqual(shocked_eps_factor(2.0, 0.9, 0.05), 0.05)

    def test_per_stock_override_wins(self):
        s = replace(STOCKS[0], revenue_elasticity={"low": 2.0, "central": 2.0, "high": 2.0})
        self.assertGreater(eps_elasticity(s, CFG)["central"], eps_elasticity(STOCKS[0], CFG)["central"])


class DrawdownTest(unittest.TestCase):
    def test_zero_cut_gives_zero_drawdown(self):
        for s in STOCKS:
            for case in CASES:
                self.assertAlmostEqual(drawdown(s, 0.0, case, CFG, DRV), 0.0, places=6)

    def test_drawdown_deepens_with_the_cut(self):
        for s in STOCKS:
            for case in CASES:
                dd = [drawdown(s, c, case, CFG, DRV) for c in CUTS]
                for a, b in zip(dd, dd[1:]):
                    self.assertLessEqual(b, a + 1e-12, (s.name, case))

    def test_drawdown_deepens_with_elasticity(self):
        s = STOCKS[0]
        lo = replace(s, revenue_elasticity={"low": 0.8, "central": 0.8, "high": 0.8})
        hi = replace(s, revenue_elasticity={"low": 1.6, "central": 1.6, "high": 1.6})
        self.assertLess(drawdown(hi, 0.2, "central", CFG, DRV), drawdown(lo, 0.2, "central", CFG, DRV))

    def test_drawdown_deepens_with_k(self):
        more_k = cfg_with(lambda r: r["revision"]["default"].update(central=0.6, high=0.6))
        for s in STOCKS:
            self.assertLess(drawdown(s, 0.2, "central", more_k, DRV), drawdown(s, 0.2, "central", CFG, DRV))

    def test_low_central_high_ordered(self):
        for s in STOCKS:
            for c in CUTS[1:]:
                lo, mid, hi = (drawdown(s, c, case, CFG, DRV) for case in CASES)
                self.assertGreaterEqual(lo, mid - 1e-12)
                self.assertGreaterEqual(mid, hi - 1e-12)

    def test_central_ten_percent_stable_across_discount_rates(self):
        for s in STOCKS:
            base = drawdown(s, 0.10, "central", CFG, DRV, discount_rate=0.10)
            for r in (0.09, 0.11, 0.12):
                dd = drawdown(s, 0.10, "central", CFG, DRV, discount_rate=r)
                self.assertLess(abs(dd - base), 0.02, (s.name, r))

    def test_credit_funded_leg_cuts_harder(self):
        neo = next(s for s in STOCKS if s.segment == "neocloud")
        self.assertLess(drawdown(neo, 0.2, "central", CFG, driver_beta(1.5)),
                        drawdown(neo, 0.2, "central", CFG, driver_beta(1.0)))

    def test_hyperscaler_leg_ignores_second_leg_beta(self):
        chip = STOCKS[0]
        self.assertAlmostEqual(drawdown(chip, 0.2, "central", CFG, driver_beta(1.5)),
                               drawdown(chip, 0.2, "central", CFG, driver_beta(1.0)))

    def test_full_cut_stays_above_total_loss(self):
        for s in STOCKS:
            self.assertGreater(drawdown(s, 1.0, "high", CFG, DRV), -1.0)

    def test_named_cuts_match_numbers(self):
        s = STOCKS[0]
        for name, c in DRV.cuts.items():
            self.assertAlmostEqual(drawdown(s, name, "central", CFG, DRV), drawdown(s, c, "central", CFG, DRV))


class TableTest(unittest.TestCase):
    def test_stock_table_has_every_stock_and_cut(self):
        t = stock_table(STOCKS, None, CFG, DRV)
        self.assertEqual(len(t), len(STOCKS) * len(DRV.cuts))
        self.assertFalse(t.isna().any().any())

    def test_headline_is_sorted_ten_percent_move(self):
        h = headline(STOCKS, CFG, DRV)
        self.assertTrue(h["move_per_10pct_cut"].is_monotonic_increasing)
        for _, row in h.iterrows():
            s = next(x for x in STOCKS if x.name == row["name"])
            self.assertAlmostEqual(row["move_per_10pct_cut"], drawdown(s, HEADLINE_CUT, "central", CFG, DRV))

    def test_partial_override_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.csv"
            df = pd.read_csv(EXAMPLES)
            df["revenue_elasticity_central"] = 1.0
            df.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                read_stocks(p)

    def test_command_line_writes_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            with redirect_stdout(io.StringIO()):
                rc = run_main(["--stocks", str(EXAMPLES), "--cut", "bust", "--cut", "0.15", "--out", d])
            self.assertEqual(rc, 0)
            t = pd.read_csv(Path(d) / "drawdowns.csv")
            self.assertEqual(len(t), len(STOCKS) * 2)
            self.assertTrue((Path(d) / "headline.csv").exists())


if __name__ == "__main__":
    unittest.main()
