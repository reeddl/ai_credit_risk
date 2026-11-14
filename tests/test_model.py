import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

from ai_pd.config import load_config
from ai_pd.defaults import (
    DefaultEvent,
    LabeledEvent,
    MarkdownKind,
    first_default,
    label_markdown,
)
from ai_pd.features import bin_score, score_features
from ai_pd.merton import merton_pd, norm_cdf
from ai_pd.pipeline import run_model
from ai_pd.scenarios import apply_scenario
from ai_pd.scorecard import inv_logit, logit, score_pd
from ai_pd.sensitivity import sensitivity_table
from ai_pd.synthetic import generate_universe

CFG = load_config()


class DefaultDefinitionTest(unittest.TestCase):
    def test_sale_at_90_percent_markdown_counts(self):
        ev = label_markdown("x", date(2026, 1, 1), 10.0, 1.0, MarkdownKind.SALE)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.event, DefaultEvent.DISTRESSED_MARKDOWN)
        self.assertAlmostEqual(ev.markdown, 0.9)

    def test_sale_at_89_percent_markdown_does_not_count(self):
        self.assertIsNone(label_markdown("x", date(2026, 1, 1), 10.0, 1.1, MarkdownKind.SALE))

    def test_down_round_with_no_sale_counts(self):
        ev = label_markdown("x", date(2026, 1, 1), 50.0, 4.0, MarkdownKind.DOWN_ROUND)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.kind, MarkdownKind.DOWN_ROUND)

    def test_down_rounds_can_be_switched_off(self):
        self.assertIsNone(
            label_markdown("x", date(2026, 1, 1), 50.0, 4.0, MarkdownKind.DOWN_ROUND, down_rounds_count=False)
        )

    def test_estimated_flag_is_kept(self):
        ev = label_markdown("x", date(2026, 1, 1), 100.0, 5.0, MarkdownKind.SALE, estimated=True)
        self.assertTrue(ev.estimated)

    def test_first_default_inside_horizon(self):
        evs = [
            LabeledEvent("a", DefaultEvent.SHUTDOWN, date(2026, 6, 1)),
            LabeledEvent("a", DefaultEvent.BANKRUPTCY_FILING, date(2026, 3, 1)),
            LabeledEvent("b", DefaultEvent.SHUTDOWN, date(2027, 6, 1)),
        ]
        first = first_default(evs, date(2026, 1, 1), 1)
        self.assertEqual(first["a"].event, DefaultEvent.BANKRUPTCY_FILING)
        self.assertNotIn("b", first)

    def test_config_matches_decided_definition(self):
        d = CFG.default_definition
        self.assertEqual(d["markdown_threshold"], 0.90)
        self.assertTrue(d["down_rounds_count"])
        self.assertEqual(len(d["events"]), 4)


class ScorecardTest(unittest.TestCase):
    def test_bin_score_directions(self):
        v = pd.Series([40, 30, 20, 10, 3, np.nan])
        self.assertEqual(list(bin_score(v, "higher_safer", [36, 24, 12, 6])), [1, 2, 3, 4, 5, 3])
        v = pd.Series([0.05, 0.2, 0.3, 0.5, 0.9])
        self.assertEqual(list(bin_score(v, "lower_safer", [0.1, 0.25, 0.4, 0.6])), [1, 2, 3, 4, 5])

    def test_neutral_scores_return_segment_prior(self):
        segs = pd.Series(list(CFG.segments))
        scores = pd.DataFrame({f: 3.0 for f in CFG.features}, index=segs.index)
        out = score_pd(segs, scores, CFG)
        for seg, pd_1y in zip(segs, out["pd_1y"]):
            self.assertAlmostEqual(pd_1y, CFG.segments[seg]["prior_pd_1y"], places=9)

    def test_riskier_compute_score_raises_pd(self):
        segs = pd.Series(["app_maker", "app_maker"])
        scores = pd.DataFrame({f: [3.0, 3.0] for f in CFG.features})
        scores.loc[1, "compute_sensitivity"] = 5.0
        out = score_pd(segs, scores, CFG)
        self.assertGreater(out["pd_1y"][1], out["pd_1y"][0])

    def test_compute_sensitivity_has_largest_weight(self):
        weights = {k: v["weight"] for k, v in CFG.features.items()}
        self.assertEqual(max(weights, key=weights.get), "compute_sensitivity")

    def test_three_year_pd_uses_constant_hazard(self):
        segs = pd.Series(["neocloud"])
        scores = pd.DataFrame({f: [3.0] for f in CFG.features})
        out = score_pd(segs, scores, CFG)
        self.assertAlmostEqual(out["pd_3y"][0], 1 - (1 - out["pd_1y"][0]) ** 3)

    def test_logit_round_trip(self):
        self.assertAlmostEqual(inv_logit(logit(0.07)), 0.07)


class MertonTest(unittest.TestCase):
    def test_no_debt_barrier_is_ten_percent_of_reference(self):
        sigma = 0.8
        pd_ = merton_pd(100.0, 100.0, 0.0, sigma, r=0.04, t=1.0, threshold=0.9)
        expected = norm_cdf(-(math.log(10.0) - 0.5 * sigma**2) / sigma)
        self.assertAlmostEqual(pd_, expected, places=9)

    def test_already_marked_down_is_certain(self):
        self.assertEqual(merton_pd(9.0, 100.0, 0.0, 0.8), 1.0)

    def test_more_debt_raises_pd(self):
        low = merton_pd(100.0, 100.0, 20.0, 0.6)
        high = merton_pd(100.0, 100.0, 200.0, 0.6)
        self.assertGreater(high, low)

    def test_lower_current_mark_raises_pd(self):
        self.assertGreater(merton_pd(40.0, 100.0, 10.0, 0.7), merton_pd(100.0, 100.0, 10.0, 0.7))


class ScenarioTest(unittest.TestCase):
    def setUp(self):
        self.df = generate_universe(CFG, n_per_segment=6, seed=11)

    def test_compute_glut_hurts_neoclouds(self):
        res = run_model(self.df, CFG).companies
        neo = res[res["segment"] == "neocloud"]
        self.assertTrue((neo["pd_1y_compute_glut"] >= neo["pd_1y"]).all())

    def test_compute_glut_lowers_app_maker_compute_cost(self):
        shocked = apply_scenario(self.df, "compute_glut", CFG)
        app = self.df["segment"] == "app_maker"
        self.assertTrue((shocked.loc[app, "compute_cost_share"] <= self.df.loc[app, "compute_cost_share"]).all())

    def test_funding_freeze_lowers_equity_marks(self):
        shocked = apply_scenario(self.df, "funding_freeze", CFG)
        self.assertTrue((shocked["equity_value_now"] < self.df["equity_value_now"]).all())

    def test_combined_is_at_least_as_bad_as_demand_stall_on_average(self):
        res = run_model(self.df, CFG).companies
        self.assertGreaterEqual(res["pd_1y_combined"].mean(), res["pd_1y_demand_stall"].mean())


class PipelineTest(unittest.TestCase):
    def test_end_to_end_is_deterministic_and_bounded(self):
        a = run_model(generate_universe(CFG, 3, seed=5), CFG).companies
        b = run_model(generate_universe(CFG, 3, seed=5), CFG).companies
        pd.testing.assert_frame_equal(a, b)
        pd_cols = [c for c in a.columns if c.startswith("pd_")]
        vals = a[pd_cols].to_numpy(dtype=float)
        self.assertTrue(np.all((vals >= 0) & (vals <= 1)))
        self.assertEqual(len(a), 3 * len(CFG.segments))

    def test_every_feature_is_scored(self):
        df = generate_universe(CFG, 2, seed=1)
        scores = score_features(df, CFG)
        self.assertEqual(set(scores.columns), set(CFG.features))
        self.assertTrue(scores.isin([1.0, 2.0, 3.0, 4.0, 5.0]).all().all())

    def test_sensitivity_covers_weights_and_priors(self):
        df = generate_universe(CFG, 2, seed=1)
        table = sensitivity_table(df, CFG)
        self.assertEqual(len(table), len(CFG.features) + len(CFG.segments))
        self.assertTrue((table["swing"].diff().dropna() <= 1e-12).all())


if __name__ == "__main__":
    unittest.main()
