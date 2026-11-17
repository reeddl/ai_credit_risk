"""Growth revision fit on analog drawdowns (ai_eq.revision, ai_eq.calibrate)."""

import io
import math
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from ai_eq.calibrate import main as calibrate_main
from ai_eq.config import DEFAULT_EQUITY, EquityConfig, load_equity_config
from ai_eq.drawdown import Valuation, drawdown
from ai_eq.inputs import read_stocks
from ai_eq.rdcf import implied_growth, value
from ai_eq.revision import FIGURES, ROLES, STATUSES, fit, growth_cut, load_episodes, load_sources
from ai_risk.driver import load_driver

ROOT = Path(__file__).resolve().parent.parent
CFG = load_equity_config()
EPISODES = load_episodes()
SOURCES = load_sources()


def simulated(kappas, floor_pe=None):
    v = CFG.valuation
    kw = dict(years_to_first_flow=1.0, explicit_years=4, fade_years=5, terminal_growth=v["terminal_growth"])
    rows = []
    for i, (kap, g0, x) in enumerate(kappas):
        e0 = 10.0
        p0 = value(e0, g0, v["discount_rate"], **kw)
        e1 = e0 * (1 - x)
        p1 = value(e1, g0 - kap * x, v["discount_rate"], **kw)
        rows.append(dict(episode=f"sim_{i}", role="kappa", peak_price=p0, peak_eps=e0,
                         later_price=p1, later_eps=e1, years_to_first_flow=1.0))
    if floor_pe:
        rows.append(dict(episode="sim_floor", role="floor", peak_price=floor_pe * 8, peak_eps=8.0,
                         later_price=floor_pe * 6, later_eps=6.0, years_to_first_flow=1.0))
    return pd.DataFrame(rows)


class DataTest(unittest.TestCase):
    def test_roles_and_reasons(self):
        self.assertTrue(set(EPISODES["role"]) <= set(ROLES))
        for _, row in EPISODES[EPISODES["role"] == "excluded"].iterrows():
            self.assertIsInstance(row["exclude_reason"], str)

    def test_every_figure_used_in_the_fit_has_a_source_row(self):
        for _, row in EPISODES[EPISODES["role"] != "excluded"].iterrows():
            for f in FIGURES:
                src = SOURCES[(SOURCES["episode"] == row["episode"]) & (SOURCES["field"] == f)]
                self.assertEqual(len(src), 1, (row["episode"], f))
                self.assertAlmostEqual(float(src["value"].iloc[0]), float(row[f]), places=6)

    def test_source_rows_say_what_they_are(self):
        self.assertTrue(set(SOURCES["status"]) <= set(STATUSES))
        for _, s in SOURCES.iterrows():
            if s["status"] in ("sourced", "derived"):
                self.assertTrue(str(s["url"]).startswith("https://"), s.to_dict())
                pd.Timestamp(s["as_of"])
            else:
                self.assertIsInstance(s["note"], str, s.to_dict())

    def test_no_future_dates(self):
        today = pd.Timestamp("2026-11-17")
        for col in ("peak_date", "later_date"):
            for d in EPISODES[col].dropna():
                self.assertLessEqual(pd.Timestamp(d), today)
        for d in SOURCES["as_of"]:
            self.assertLessEqual(pd.Timestamp(d), today)


class FitTest(unittest.TestCase):
    def test_fit_recovers_a_known_kappa(self):
        result = fit(simulated([(0.8, 0.25, 0.3)]), CFG)
        self.assertAlmostEqual(result["episodes"][0].kappa, 0.8, places=6)

    def test_band_is_min_median_max(self):
        result = fit(simulated([(0.4, 0.2, 0.2), (0.9, 0.3, 0.3), (1.5, 0.35, 0.15)], floor_pe=6.0), CFG)
        band = result["kappa"]
        self.assertAlmostEqual(band["low"], 0.4, places=6)
        self.assertAlmostEqual(band["central"], 0.9, places=6)
        self.assertAlmostEqual(band["high"], 1.5, places=6)
        self.assertAlmostEqual(result["floor_pe"], 6.0)

    def test_config_matches_the_fit(self):
        result = fit(EPISODES, CFG)
        for case, val in result["kappa"].items():
            self.assertAlmostEqual(CFG.kappa("chip_designer")[case], val, delta=0.005)
        self.assertAlmostEqual(CFG.floor_pe, result["floor_pe"], delta=0.05)

    def test_growth_cut_capped_at_floor_and_zero_for_a_rise(self):
        self.assertAlmostEqual(growth_cut(0.1, 1.0, 0.5, -0.1), 0.2)
        self.assertAlmostEqual(growth_cut(0.1, 1.0, 0.1, -0.1), 0.1)
        self.assertEqual(growth_cut(0.1, 1.0, -0.2, -0.1), 0.0)
        self.assertEqual(growth_cut(-0.2, 1.0, 0.3, -0.1), 0.0)

    def test_drawdown_never_below_the_floor_multiple(self):
        stock = read_stocks(ROOT / "tests" / "data" / "equity_example_stocks.csv")[0]
        drv = load_driver()
        val = Valuation.for_stock(stock, CFG)
        for c in (0.2, 0.5, 1.0):
            dd = drawdown(stock, c, "high", CFG, drv)
            e = stock.eps_next * max(1 - 1.4 * (0.70 + 0.15) / 0.55 * c, CFG.valuation["earnings_floor"])
            g_floor = implied_growth(CFG.floor_pe, 1.0, val.r, **val.kw)
            self.assertGreaterEqual(dd + 1e-9, value(e, g_floor, val.r, **val.kw) / stock.price - 1)

    def test_command_line_writes_report(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fit.md"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(calibrate_main(["--out", str(out)]), 0)
            self.assertIn("kappa: low", out.read_text())


if __name__ == "__main__":
    unittest.main()
