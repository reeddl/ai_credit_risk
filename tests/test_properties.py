"""Property and sanity tests for the scorecard, Merton and scenario layers.

Property tests draw many random inputs from a seeded generator, so they are
repeatable without extra dependencies. Each test states the property it
checks in its name.
"""

import math
import unittest

import numpy as np
import pandas as pd

from ai_pd.calibration import implied_grade
from ai_pd.config import ModelConfig, load_config, validate
from ai_pd.features import RAW_COLUMNS, bin_score, compute_metrics, score_features
from ai_pd.merton import equity_value, markdown_barrier, merton_pd
from ai_pd.pipeline import run_model, segment_summary
from ai_pd.scenarios import apply_scenario, severities
from ai_pd.scorecard import inv_logit, logit, pd_3y_from_1y, score_pd
from ai_pd.synthetic import generate_universe

CFG = load_config()
RNG_SEED = 20261114
N_DRAWS = 300


def _universe(n=8, seed=11):
    return generate_universe(CFG, n_per_segment=n, seed=seed)


class ScorecardPropertyTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(RNG_SEED)
        self.segs = list(CFG.segments)
        self.feats = list(CFG.features)

    def _random_scores(self, n):
        return pd.DataFrame({f: self.rng.integers(1, 6, n).astype(float) for f in self.feats})

    def test_pd_always_strictly_between_zero_and_one(self):
        segs = pd.Series(self.rng.choice(self.segs, N_DRAWS))
        out = score_pd(segs, self._random_scores(N_DRAWS), CFG)
        self.assertTrue(((out["pd_1y"] > 0) & (out["pd_1y"] < 1)).all())
        self.assertTrue(((out["pd_3y"] >= out["pd_1y"]) & (out["pd_3y"] < 1)).all())

    def test_pd_rises_with_every_feature_score(self):
        segs = pd.Series(self.rng.choice(self.segs, N_DRAWS))
        base = self._random_scores(N_DRAWS)
        base_pd = score_pd(segs, base, CFG)["pd_1y"]
        for f in self.feats:
            worse = base.copy()
            worse[f] = (worse[f] + 1).clip(upper=5)
            moved = worse[f] > base[f]
            new_pd = score_pd(segs, worse, CFG)["pd_1y"]
            self.assertTrue((new_pd[moved] > base_pd[moved]).all(), f)
            self.assertTrue(np.allclose(new_pd[~moved], base_pd[~moved]), f)

    def test_every_weight_is_positive(self):
        for f, spec in CFG.features.items():
            self.assertGreater(spec["weight"], 0, f)

    def test_log_odds_pieces_add_up(self):
        segs = pd.Series(self.rng.choice(self.segs, 50))
        out = score_pd(segs, self._random_scores(50), CFG, {"demand_stall": 1.0})
        total = out["logodds_prior"] + out["logodds_features"] + out["logodds_scenario"]
        np.testing.assert_allclose(logit(out["pd_1y"]), total, rtol=1e-9, atol=1e-9)

    def test_raising_a_segment_prior_raises_its_pds_only(self):
        segs = pd.Series(self.rng.choice(self.segs, N_DRAWS))
        scores = self._random_scores(N_DRAWS)
        base = score_pd(segs, scores, CFG)["pd_1y"]
        cfg = CFG.copy()
        cfg.segments["neocloud"]["prior_pd_1y"] *= 2
        new = score_pd(segs, scores, cfg)["pd_1y"]
        self.assertTrue((new[segs == "neocloud"] > base[segs == "neocloud"]).all())
        self.assertTrue(np.allclose(new[segs != "neocloud"], base[segs != "neocloud"]))

    def test_scores_bounded_by_prior_and_weights(self):
        # The most any company can move from its prior is sum(|w|) * 2 log-odds.
        span = 2 * sum(f["weight"] for f in CFG.features.values())
        for seg, spec in CFG.segments.items():
            lo = inv_logit(logit(spec["prior_pd_1y"]) - span)
            hi = inv_logit(logit(spec["prior_pd_1y"]) + span)
            segs = pd.Series([seg, seg])
            scores = pd.DataFrame({f: [1.0, 5.0] for f in self.feats})
            out = score_pd(segs, scores, CFG)["pd_1y"]
            self.assertAlmostEqual(out[0], lo, places=12)
            self.assertAlmostEqual(out[1], hi, places=12)

    def test_three_year_pd_monotone_in_one_year(self):
        p = np.sort(self.rng.uniform(0, 0.99, N_DRAWS))
        self.assertTrue(np.all(np.diff(pd_3y_from_1y(p)) >= 0))


class BinScorePropertyTest(unittest.TestCase):
    def test_score_is_monotone_in_metric_for_every_binned_feature(self):
        rng = np.random.default_rng(RNG_SEED)
        for f, spec in CFG.features.items():
            if spec["direction"] == "score":
                continue
            lo, hi = min(spec["bins"]), max(spec["bins"])
            v = pd.Series(np.sort(rng.uniform(lo - abs(lo) - 1, hi * 2 + 1, 500)))
            s = bin_score(v, spec["direction"], spec["bins"]).to_numpy()
            diffs = np.diff(s)
            if spec["direction"] == "lower_safer":
                self.assertTrue(np.all(diffs >= 0), f)
            else:
                self.assertTrue(np.all(diffs <= 0), f)
            self.assertTrue(set(np.unique(s)) <= {1.0, 2.0, 3.0, 4.0, 5.0})

    def test_bin_edges_are_inclusive_on_the_safe_side(self):
        for f, spec in CFG.features.items():
            if spec["direction"] == "score":
                continue
            s = bin_score(pd.Series(spec["bins"], dtype=float), spec["direction"], spec["bins"])
            self.assertEqual(list(s), [1.0, 2.0, 3.0, 4.0], f)

    def test_missing_metric_scores_neutral(self):
        for direction, bins in (("higher_safer", [4, 3, 2, 1]), ("lower_safer", [1, 2, 3, 4]), ("score", None)):
            self.assertEqual(bin_score(pd.Series([np.nan]), direction, bins)[0], 3.0)

    def test_judgment_scores_are_clipped_to_1_to_5(self):
        self.assertEqual(list(bin_score(pd.Series([-2.0, 0.4, 7.0]), "score", None)), [1.0, 1.0, 5.0])


class FeatureSanityTest(unittest.TestCase):
    def setUp(self):
        self.df = _universe()

    def test_cash_generating_company_gets_best_runway_score(self):
        df = self.df.copy()
        df["monthly_net_burn"] = -1e6
        self.assertTrue((score_features(df, CFG)["runway"] == 1).all())

    def test_more_cash_never_hurts(self):
        more = self.df.copy()
        more["cash"] *= 3
        base = run_model(self.df, CFG).companies["pd_1y"]
        new = run_model(more, CFG).companies["pd_1y"]
        self.assertTrue((new <= base + 1e-12).all())

    def test_more_debt_never_helps(self):
        more = self.df.copy()
        more["debt"] *= 2
        more["debt_due_24m"] *= 2
        base = run_model(self.df, CFG).companies["pd_1y"]
        new = run_model(more, CFG).companies["pd_1y"]
        self.assertTrue((new >= base - 1e-12).all())

    def test_scaling_all_money_leaves_scores_unchanged_except_labor_intensity(self):
        money = [
            "cash", "monthly_net_burn", "revenue_annual", "debt", "debt_due_24m", "total_assets",
            "hard_assets", "equity_value_ref", "equity_value_now",
        ]
        scaled = self.df.copy()
        scaled[money] *= 7.0
        a = score_features(self.df, CFG).drop(columns="labor_intensity")
        b = score_features(scaled, CFG).drop(columns="labor_intensity")
        pd.testing.assert_frame_equal(a, b)

    def test_zero_revenue_company_has_neutral_margin_and_customer_scores(self):
        df = self.df.copy()
        df["revenue_annual"] = 0.0
        df["revenue_stage"] = "none"
        m = compute_metrics(df)
        self.assertTrue(m["gross_margin"].isna().all())
        s = score_features(df, CFG)
        self.assertTrue((s[["gross_margin", "customer_concentration", "counterparty_quality"]] == 3).all().all())
        self.assertTrue((s["revenue"] == 5).all())

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_metrics(self.df.drop(columns="cash"))

    def test_raw_columns_are_unique(self):
        self.assertEqual(len(RAW_COLUMNS), len(set(RAW_COLUMNS)))


class MertonPropertyTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(RNG_SEED)

    def _draw(self):
        ref = self.rng.uniform(10, 1000)
        return dict(
            equity_now=ref * self.rng.uniform(0.15, 1.5),
            equity_ref=ref,
            debt=self.rng.uniform(0, 2 * ref),
            equity_vol=self.rng.uniform(0.2, 1.5),
        )

    def test_pd_in_unit_interval(self):
        for _ in range(N_DRAWS):
            p = merton_pd(**self._draw())
            self.assertTrue(0.0 <= p <= 1.0)

    def test_pd_rises_with_volatility(self):
        for _ in range(100):
            d = self._draw()
            lo = merton_pd(**{**d, "equity_vol": d["equity_vol"]})
            hi = merton_pd(**{**d, "equity_vol": d["equity_vol"] * 1.3})
            self.assertGreaterEqual(hi, lo - 1e-12)

    def test_pd_falls_as_current_mark_rises(self):
        for _ in range(100):
            d = self._draw()
            lo = merton_pd(**{**d, "equity_now": d["equity_now"] * 1.2})
            hi = merton_pd(**d)
            self.assertLessEqual(lo, hi + 1e-12)

    def test_equity_value_is_increasing_in_assets_and_below_assets(self):
        v = np.linspace(1, 500, 200)
        e = np.array([equity_value(x, 100.0, 0.5, 0.04, 1.0) for x in v])
        self.assertTrue(np.all(np.diff(e) > 0))
        self.assertTrue(np.all(e <= v + 1e-9))
        self.assertTrue(np.all(e >= np.maximum(v - 100.0, 0) - 1e-9))

    def test_markdown_barrier_hits_its_target(self):
        for _ in range(50):
            ref, debt, sig = self.rng.uniform(50, 500), self.rng.uniform(1, 500), self.rng.uniform(0.2, 1.0)
            vstar = markdown_barrier(ref, debt, sig, 0.04, 1.0, 0.9)
            self.assertAlmostEqual(equity_value(vstar, debt, sig, 0.04, 1.0), 0.1 * ref, delta=1e-6 * ref)

    def test_higher_markdown_threshold_lowers_pd(self):
        # A stricter (deeper) markdown test is harder to hit.
        for _ in range(50):
            d = self._draw()
            if d["equity_now"] <= 0.2 * d["equity_ref"]:
                continue
            self.assertLessEqual(merton_pd(**d, threshold=0.95), merton_pd(**d, threshold=0.80) + 1e-12)

    def test_missing_inputs_give_nan(self):
        self.assertTrue(math.isnan(merton_pd(10, 0, 5, 0.5)))
        self.assertTrue(math.isnan(merton_pd(10, 10, 5, 0.0)))


class ScenarioOrderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _universe(n=20, seed=3)
        cls.res = run_model(cls.df, CFG).companies

    def test_demand_stall_funding_freeze_and_rates_up_never_lower_pd(self):
        for s in ("demand_stall", "funding_freeze", "rates_up"):
            self.assertTrue((self.res[f"pd_1y_{s}"] >= self.res["pd_1y"] - 1e-12).all(), s)

    def test_combined_never_lower_than_base(self):
        self.assertTrue((self.res["pd_1y_combined"] >= self.res["pd_1y"] - 1e-12).all())

    def test_combined_at_least_as_bad_as_any_single_scenario_for_compute_sellers(self):
        # For companies that buy compute, a compute glut lowers costs, so the
        # combined case can come in under demand stall alone. For sellers of
        # compute every part of the combined case hurts.
        sellers = self.df["compute_revenue_share"] > 0
        worst_single = self.res[[f"pd_1y_{s}" for s in severities("combined", CFG)]].max(axis=1)
        self.assertTrue((self.res.loc[sellers, "pd_1y_combined"] >= worst_single[sellers] - 1e-12).all())

    def test_combined_mean_exceeds_every_single_scenario_mean(self):
        for s in severities("combined", CFG):
            self.assertGreater(self.res["pd_1y_combined"].mean(), self.res[f"pd_1y_{s}"].mean(), s)

    def test_compute_glut_hits_gpu_lessors_and_neoclouds_hardest(self):
        rise = self.res["pd_1y_compute_glut"] - self.res["pd_1y"]
        by_seg = rise.groupby(self.res["segment"]).mean()
        top_two = set(by_seg.nlargest(2).index)
        self.assertEqual(top_two, {"neocloud", "gpu_lessor"})

    def test_rates_up_hurts_floating_rate_borrowers_more(self):
        df = self.df[self.df["debt"] > 0].copy()
        hi, lo = df.copy(), df.copy()
        hi["floating_debt_share"], lo["floating_debt_share"] = 1.0, 0.0
        hi_s = apply_scenario(hi, "rates_up", CFG)
        lo_s = apply_scenario(lo, "rates_up", CFG)
        self.assertTrue((hi_s["monthly_net_burn"] >= lo_s["monthly_net_burn"]).all())

    def test_funding_freeze_raises_merton_pd(self):
        ok = self.res["pd_merton_1y"].notna()
        self.assertTrue((self.res.loc[ok, "pd_merton_1y_funding_freeze"] >= self.res.loc[ok, "pd_merton_1y"] - 1e-12).all())

    def test_scenarios_do_not_mutate_input(self):
        before = self.df.copy()
        for s in CFG.scenarios:
            apply_scenario(self.df, s, CFG)
        pd.testing.assert_frame_equal(self.df, before)

    def test_combined_severities_sum_parts(self):
        self.assertEqual(severities("combined", CFG), {k: 1.0 for k in ("compute_glut", "demand_stall", "funding_freeze", "rates_up")})


class SegmentSanityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = _universe(n=30, seed=5)
        cls.summary = segment_summary(run_model(df, CFG), CFG)

    def test_established_segments_safer_than_frontier_segments(self):
        established = ["power", "fab_equipment", "chip_designer", "networking", "data_strong_moat"]
        frontier = ["app_maker", "robotics", "open_weight_model", "neocloud"]
        self.assertLess(self.summary.loc[established, "pd_1y"].max(), self.summary.loc[frontier, "pd_1y"].min())

    def test_strong_moat_data_safer_than_weak_moat_data(self):
        self.assertLess(self.summary.loc["data_strong_moat", "pd_1y"], self.summary.loc["data_weak_moat", "pd_1y"])

    def test_segment_means_sit_on_the_known_side_of_priors(self):
        # Known v1 design point: the prior is the PD at all-neutral scores, so
        # cash-generating, low-debt segments land well below their prior.
        # This test pins that behaviour so a recentring change is deliberate.
        for seg in ("power", "fab_equipment", "chip_designer"):
            self.assertLess(self.summary.loc[seg, "pd_1y"], self.summary.loc[seg, "prior_pd_1y"])


class ConfigValidationTest(unittest.TestCase):
    def _bad(self, mutate):
        cfg = CFG.copy()
        mutate(cfg.raw)
        with self.assertRaises(ValueError):
            validate(ModelConfig(cfg.raw))

    def test_rejects_prior_outside_unit_interval(self):
        self._bad(lambda r: r["segments"]["power"].__setitem__("prior_pd_1y", 1.2))

    def test_rejects_unordered_bins(self):
        self._bad(lambda r: r["features"]["runway"].__setitem__("bins", [6, 12, 24, 36]))

    def test_rejects_wrong_bin_count(self):
        self._bad(lambda r: r["features"]["leverage"].__setitem__("bins", [0.1, 0.3, 0.5]))

    def test_rejects_unknown_direction(self):
        self._bad(lambda r: r["features"]["moat"].__setitem__("direction", "sideways"))

    def test_rejects_nested_combined_scenario(self):
        self._bad(lambda r: r["scenarios"].__setitem__("worse", {"combine": ["combined"]}))

    def test_rejects_decreasing_grade_bounds(self):
        self._bad(lambda r: r["calibration"].__setitem__("upper_pd_1y", [0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]))


class CalibrationTest(unittest.TestCase):
    def test_grades_are_monotone_in_pd(self):
        grades = CFG.calibration["grades"]
        p = pd.Series(np.logspace(-5, -0.01, 400))
        idx = implied_grade(p, CFG).map(grades.index)
        self.assertTrue((idx.diff().dropna() >= 0).all())

    def test_boundary_pd_gets_the_safer_grade(self):
        bounds = CFG.calibration["upper_pd_1y"]
        g = implied_grade(pd.Series([bounds[0], bounds[3]]), CFG)
        self.assertEqual(list(g), [CFG.calibration["grades"][0], CFG.calibration["grades"][3]])


if __name__ == "__main__":
    unittest.main()
