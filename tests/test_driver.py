"""Shared compute demand driver and the credit scenarios wired to it."""

import copy
import io
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from ai_pd.config import ModelConfig, load_config, validate
from ai_pd.pipeline import run_model
from ai_pd.run import main as run_main
from ai_pd.scenarios import SHOCK_FLOOR, apply_scenario, scaled_scenario, severities
from ai_pd.synthetic import generate_universe
from ai_risk.driver import DEFAULT_DRIVER, Driver, load_driver, validate_driver

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
DRV = load_driver()
GOLDEN = ROOT / "tests" / "data" / "golden_pd_before_driver.csv"
REAL = ROOT / "benchmark" / "data" / "companies_real.csv"
CUTS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]
LINKED = [n for n, sc in CFG.scenarios.items() if "driver_leg" in sc]
UNLINKED = [n for n, sc in CFG.scenarios.items() if "driver_leg" not in sc and "combine" not in sc]


def driver_with(**changes) -> Driver:
    with open(DEFAULT_DRIVER, "rb") as fh:
        raw = tomllib.load(fh)
    for path, value in changes.items():
        node = raw["driver"]
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return Driver(raw)


class DriverTest(unittest.TestCase):
    def test_default_file_loads_and_is_sourced(self):
        d = DRV.raw["driver"]
        self.assertGreater(DRV.consensus, DRV.prior_year)
        for key in ("as_of", "source", "url"):
            self.assertTrue(d[key])

    def test_resolve(self):
        self.assertEqual(DRV.resolve(None), DRV.reference_cut)
        self.assertEqual(DRV.resolve("bust"), DRV.cuts["bust"])
        self.assertEqual(DRV.resolve(0.15), 0.15)
        with self.assertRaises(ValueError):
            DRV.resolve("crash")
        with self.assertRaises(ValueError):
            DRV.resolve(1.2)
        with self.assertRaises(ValueError):
            DRV.resolve(-0.1)

    def test_zero_cut_returns_consensus(self):
        self.assertEqual(DRV.level(0.0), DRV.consensus)
        for leg in ("hyperscaler", "credit_funded"):
            self.assertEqual(DRV.leg_cut(0.0, leg), 0.0)
            self.assertEqual(DRV.shock_scale(0.0, leg), 0.0)

    def test_named_cuts_are_ordered(self):
        c = DRV.cuts
        self.assertLess(c["mild"], c["stall"])
        self.assertLess(c["stall"], c["bust"])

    def test_second_leg_never_cuts_less_and_caps_at_one(self):
        for c in CUTS:
            l1, l2 = DRV.leg_cut(c, "hyperscaler"), DRV.leg_cut(c, "credit_funded")
            self.assertGreaterEqual(l2, l1)
            self.assertLessEqual(l2, 1.0)

    def test_level_and_growth_fall_with_the_cut(self):
        levels = [DRV.level(c) for c in CUTS]
        self.assertEqual(levels, sorted(levels, reverse=True))
        self.assertAlmostEqual(DRV.growth(0.0), DRV.consensus / DRV.prior_year - 1)

    def test_reference_cut_scale_is_one(self):
        for leg in ("hyperscaler", "credit_funded"):
            self.assertAlmostEqual(DRV.shock_scale(None, leg), 1.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            validate_driver(driver_with(leg2__beta=0.8))
        with self.assertRaises(ValueError):
            validate_driver(driver_with(credit_reference__cut=0.0))
        with self.assertRaises(ValueError):
            validate_driver(driver_with(cuts={"mild": 1.5}))
        with self.assertRaises(ValueError):
            DRV.leg_cut(0.1, "sovereign")

    def test_unknown_driver_leg_in_model_config_fails(self):
        raw = copy.deepcopy(CFG.raw)
        raw["scenarios"]["demand_stall"]["driver_leg"] = "sovereign"
        with self.assertRaises(ValueError):
            validate(ModelConfig(raw))


class ReferenceReproducesVersionOneTest(unittest.TestCase):
    """At the reference cut every PD matches the outputs saved before the driver."""

    @classmethod
    def setUpClass(cls):
        cls.golden = pd.read_csv(GOLDEN)
        cls.universes = {
            "synthetic": generate_universe(CFG, 5, 7),
            "real": pd.read_csv(REAL),
        }

    def check(self, cut):
        for tag, df in self.universes.items():
            got = run_model(df, CFG, cut, DRV).companies.set_index("company_id")
            want = self.golden[self.golden["universe"] == tag].set_index("company_id")
            cols = [c for c in want.columns if c.startswith("pd_")]
            self.assertEqual(sorted(got.index), sorted(want.index))
            diff = (got.loc[want.index, cols] - want[cols]).abs().max().max()
            self.assertLess(diff, 1e-9, f"{tag} at cut {cut}: max PD difference {diff}")

    def test_default_cut(self):
        self.check(None)

    def test_reference_cut_given_by_number(self):
        self.check(DRV.reference_cut)

    def test_reference_cut_given_by_name(self):
        self.check("stall")


class ScenarioScalingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_universe(CFG, 5, 7)

    def test_zero_cut_leaves_linked_scenarios_inert(self):
        for name in LINKED:
            shocked = apply_scenario(self.df, name, CFG, 0.0, DRV)
            pd.testing.assert_frame_equal(shocked, self.df)
            self.assertEqual(set(severities(name, CFG, 0.0, DRV).values()), {0.0})

    def test_unlinked_scenarios_do_not_move_with_the_cut(self):
        for name in UNLINKED:
            base = apply_scenario(self.df, name, CFG, None, DRV)
            for c in CUTS:
                pd.testing.assert_frame_equal(apply_scenario(self.df, name, CFG, c, DRV), base)
                self.assertEqual(severities(name, CFG, c, DRV), severities(name, CFG, None, DRV))

    def test_shocks_scale_in_proportion_and_respect_the_floor(self):
        for name in LINKED:
            leg = CFG.scenarios[name]["driver_leg"]
            written = CFG.scenarios[name]
            for c in CUTS:
                sc = scaled_scenario(name, CFG, c, DRV)
                k = DRV.shock_scale(c, leg)
                self.assertAlmostEqual(sc["severity"], written.get("severity", 1.0) * k)
                for key in ("gpu_price_change", "utilisation_change", "revenue_change", "growth_change"):
                    if key in written:
                        self.assertAlmostEqual(sc[key], max(written[key] * k, SHOCK_FLOOR))
                        self.assertGreaterEqual(sc[key], SHOCK_FLOOR)

    def test_full_cut_keeps_inputs_valid(self):
        shocked = apply_scenario(self.df, "combined", CFG, 1.0, DRV)
        self.assertTrue((shocked["revenue_annual"] >= 0).all())
        self.assertTrue(np.isfinite(shocked.select_dtypes("number")).all().all())
        res = run_model(self.df, CFG, 1.0, DRV).companies
        pds = res[[c for c in res.columns if c.startswith("pd_")]]
        self.assertTrue(((pds >= 0) & (pds <= 1)).all().all())

    def test_demand_stall_pd_rises_with_the_cut(self):
        prev = None
        for c in CUTS:
            pd_c = run_model(self.df, CFG, c, DRV).companies["pd_1y_demand_stall"]
            if prev is not None:
                self.assertTrue((pd_c >= prev - 1e-12).all(), f"demand stall PD fell at cut {c}")
            prev = pd_c

    def test_compute_glut_pd_rises_with_the_cut_for_pure_sellers(self):
        # Buyers of compute gain from cheaper GPUs, so only sellers must rise.
        sellers = (self.df["compute_cost_share"] == 0) & (self.df["compute_revenue_share"] > 0)
        self.assertTrue(sellers.any())
        prev = None
        for c in CUTS:
            pd_c = run_model(self.df, CFG, c, DRV).companies.loc[sellers, "pd_1y_compute_glut"]
            if prev is not None:
                self.assertTrue((pd_c >= prev - 1e-12).all(), f"compute glut PD fell for a seller at cut {c}")
            prev = pd_c

    def test_bust_is_harsher_than_mild_on_average(self):
        mild = run_model(self.df, CFG, "mild", DRV).companies["pd_1y_combined"].mean()
        bust = run_model(self.df, CFG, "bust", DRV).companies["pd_1y_combined"].mean()
        self.assertGreater(bust, mild)


class CommandLineTest(unittest.TestCase):
    def test_named_and_numeric_cuts_run(self):
        for cut in ("bust", "0.15"):
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                run_main(["--synthetic", "--n-per-segment", "2", "--compute-cut", cut, "--out", tmp])
                self.assertTrue(any(Path(tmp).iterdir()))


if __name__ == "__main__":
    unittest.main()
