import math
import unittest

import numpy as np
import pandas as pd

from ai_pd.market_implied import (
    hazard_from_spread,
    implied_pd,
    implied_pd_frame,
    kendall_tau_b,
    physical_pd_estimate,
    rn_to_physical_ratio,
    spearman,
    spearman_permutation_pvalue,
    spread_from_pd,
)


class CreditTriangleTest(unittest.TestCase):
    def test_textbook_value(self):
        # 100 bps at 40% recovery: hazard 1/60 per year.
        self.assertAlmostEqual(hazard_from_spread(100, 0.4), 0.01 / 0.6)
        self.assertAlmostEqual(implied_pd(100, 1.0, 0.4), 1 - math.exp(-0.01 / 0.6))

    def test_coreweave_order_of_magnitude(self):
        # Press reported ~1,000 bps as roughly a 64% five-year default chance.
        self.assertAlmostEqual(implied_pd(1000, 5.0, 0.4), 0.565, delta=0.01)
        self.assertAlmostEqual(implied_pd(1000, 5.0, 0.2), 0.465, delta=0.01)

    def test_zero_spread_zero_pd(self):
        self.assertEqual(implied_pd(0.0), 0.0)

    def test_monotone_in_spread_recovery_and_horizon(self):
        s = np.linspace(0, 3000, 200)
        p = [implied_pd(x) for x in s]
        self.assertTrue(np.all(np.diff(p) > 0))
        self.assertLess(implied_pd(200, 1, 0.2), implied_pd(200, 1, 0.6))
        self.assertLess(implied_pd(200, 1), implied_pd(200, 5))

    def test_round_trip(self):
        for s in (5, 50, 288, 1155, 4000):
            self.assertAlmostEqual(spread_from_pd(implied_pd(s, 1.0), 1.0), s, places=6)

    def test_bounds(self):
        for s in (1, 100, 1_000, 5_000):
            p = implied_pd(s, 5.0)
            self.assertTrue(0 <= p < 1)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            hazard_from_spread(-1)
        with self.assertRaises(ValueError):
            hazard_from_spread(100, 1.0)
        with self.assertRaises(ValueError):
            spread_from_pd(1.0)


class PhysicalPDTest(unittest.TestCase):
    def test_physical_below_risk_neutral(self):
        for s in (20, 80, 200, 500, 1500):
            self.assertLess(physical_pd_estimate(s), implied_pd(s))

    def test_ratio_falls_as_credit_worsens(self):
        r = [rn_to_physical_ratio(s) for s in (10, 100, 200, 400, 2000)]
        self.assertEqual(r, sorted(r, reverse=True))
        self.assertGreaterEqual(min(r), 1.0)

    def test_physical_pd_still_rises_with_spread_across_band_edges(self):
        s = np.linspace(1, 3000, 3000)
        p = np.array([physical_pd_estimate(x) for x in s])
        # A step down in the ratio can only raise the PD.
        self.assertTrue(np.all(np.diff(p) >= -1e-15))

    def test_frame_adds_columns(self):
        q = pd.DataFrame({"company_id": ["a", "b"], "spread_bps": [100.0, 400.0]})
        out = implied_pd_frame(q)
        for c in ("pd_rn_1y", "pd_rn_5y", "pd_physical_1y_est", "hazard_rn"):
            self.assertIn(c, out)
        self.assertTrue((out["pd_rn_5y"] > out["pd_rn_1y"]).all())


class RankStatsTest(unittest.TestCase):
    def test_spearman_perfect_and_reversed(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_spearman_known_value(self):
        # Classic example: rho = 1 - 6*sum(d^2)/(n(n^2-1)) with d = [0, -1, 1, 0, 0].
        self.assertAlmostEqual(spearman([1, 2, 3, 4, 5], [1, 3, 2, 4, 5]), 0.9)

    def test_spearman_handles_ties_and_constant(self):
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [1, 2, 2, 3]), 5 / 6, places=9)
        self.assertTrue(math.isnan(spearman([1, 1, 1], [1, 2, 3])))

    def test_spearman_invariant_to_monotone_transform(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=30), rng.normal(size=30)
        self.assertAlmostEqual(spearman(x, y), spearman(np.exp(x), y ** 3))

    def test_kendall_known_values(self):
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3, 4, 5], [1, 3, 2, 4, 5]), 0.8)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [3, 2, 1]), -1.0)

    def test_permutation_pvalue(self):
        self.assertLess(spearman_permutation_pvalue(range(12), range(12), n_perm=2000), 0.01)
        rng = np.random.default_rng(3)
        p = spearman_permutation_pvalue(rng.normal(size=12), rng.normal(size=12), n_perm=2000)
        self.assertTrue(0 < p <= 1)


if __name__ == "__main__":
    unittest.main()
