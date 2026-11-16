import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from ai_pd.config import load_config
from ai_pd.features import RAW_COLUMNS
from ai_pd.pipeline import run_model

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "cases" / "mercor" / "run_mercor.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_mercor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MercorCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()
        cls.cfg = load_config()
        cls.df, cls.table = cls.mod.load_inputs()

    def test_inputs_cover_raw_columns(self):
        self.assertEqual(list(self.df.columns), RAW_COLUMNS)
        self.assertEqual(len(self.df), 1)

    def test_every_figure_has_a_source_or_reason(self):
        for r in self.table.itertuples():
            if r.kind in ("reported", "derived"):
                self.assertTrue(r.source and r.url, f"{r.field} needs a source and URL")
            if r.kind in ("estimate",):
                self.assertTrue(r.note, f"{r.field} needs a stated assumption")

    def test_segment_is_known(self):
        self.assertIn(self.df["segment"].iloc[0], self.cfg.segments)

    def test_pd_in_range_and_stress_raises_it(self):
        c = run_model(self.df, self.cfg).companies.iloc[0]
        self.assertTrue(0 < c["pd_1y"] < 1)
        for name in ("demand_stall", "funding_freeze", "combined"):
            self.assertGreater(c[f"pd_1y_{name}"], c["pd_1y"])

    def test_losing_revenue_raises_pd(self):
        base = self.mod.base_pd(self.df, self.cfg)
        small, big = (self.mod.lose_a_lab(self.df, self.cfg, s) for s in self.mod.LOSS_SHARES)
        self.assertGreater(small["pd_1y"], base)
        self.assertGreater(big["pd_1y"], small["pd_1y"])
        self.assertGreater(big["new_monthly_burn"], small["new_monthly_burn"])


if __name__ == "__main__":
    unittest.main()
