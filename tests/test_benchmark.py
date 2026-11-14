"""Data integrity and end-to-end tests for the CDS benchmark."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai_pd.benchmark import (
    DATA_DIR,
    SNAPSHOTS,
    compare,
    load_data,
    rank_stats,
    segment_gaps,
    select_quotes,
    write_report,
)
from ai_pd.config import load_config
from ai_pd.features import RAW_COLUMNS, check_columns

CFG = load_config()
QUALITY = {"provider_summary", "wire_report", "press_summary", "press_unsourced", "bank_research"}
KINDS = {"filing", "market", "judgment", "derived"}
LATEST_ALLOWED = pd.Timestamp("2026-11-14")


class QuoteDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.q = cls.data.quotes

    def test_every_quote_is_dated_sourced_and_linked(self):
        for col in ("as_of", "data_provider", "read_at", "url", "quality"):
            self.assertFalse(self.q[col].isna().any(), col)
        self.assertTrue(self.q["url"].str.startswith("https://").all())
        self.assertTrue(set(self.q["quality"]) <= QUALITY)

    def test_quotes_are_not_from_the_future(self):
        self.assertTrue((self.q["as_of"] <= LATEST_ALLOWED).all())

    def test_spreads_are_plausible(self):
        self.assertTrue(self.q["spread_bps"].between(1, 5000).all())
        self.assertTrue(set(self.q["bound"]) <= {"point", "lower"})
        self.assertTrue((self.q["tenor_years"] == 5).all())

    def test_no_duplicate_quotes(self):
        self.assertFalse(self.q.duplicated(["company_id", "as_of", "spread_bps"]).any())

    def test_select_latest_takes_the_newest_in_window(self):
        sel = select_quotes(self.q, *SNAPSHOTS["latest"]).set_index("company_id")
        self.assertEqual(sel.loc["oracle", "as_of"], pd.Timestamp("2026-11-06"))
        summer = select_quotes(self.q, *SNAPSHOTS["summer_2026"])
        self.assertTrue((summer["as_of"] <= pd.Timestamp("2026-08-31")).all())
        self.assertEqual(sel.index.is_unique, True)


class CompanyDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.c = cls.data.companies
        cls.src = pd.read_csv(DATA_DIR / "companies_sources.csv")

    def test_inputs_have_every_model_column_and_no_gaps(self):
        check_columns(self.c)
        self.assertFalse(self.c[RAW_COLUMNS].isna().any().any())

    def test_every_company_is_mapped_quoted_and_valid(self):
        ids = set(self.c["company_id"])
        self.assertEqual(ids, set(self.data.segment_map["company_id"]))
        self.assertTrue(ids <= set(self.data.quotes["company_id"]))
        self.assertTrue(set(self.c["segment"]) <= set(CFG.segments))
        merged = self.c.merge(self.data.segment_map, on="company_id", suffixes=("", "_map"))
        self.assertTrue((merged["segment"] == merged["segment_map"]).all())

    def test_every_input_has_a_provenance_row(self):
        need = [c for c in RAW_COLUMNS if c not in ("company_id", "segment", "public")]
        have = set(zip(self.src["company_id"], self.src["field"]))
        missing = [(cid, f) for cid in self.c["company_id"] for f in need if (cid, f) not in have]
        self.assertEqual(missing, [])

    def test_provenance_rows_are_complete(self):
        self.assertTrue(set(self.src["kind"]) <= KINDS)
        sourced = self.src[self.src["kind"].isin(["filing", "market"])]
        self.assertFalse(sourced[["source", "url", "date"]].isna().any().any())
        judged = self.src[self.src["kind"] == "judgment"]
        self.assertFalse(judged["note"].isna().any())

    def test_balance_sheet_identities(self):
        c = self.c
        self.assertTrue((c["cash"] >= 0).all())
        self.assertTrue((c["hard_assets"] <= c["total_assets"]).all())
        self.assertTrue((c["cash"] <= c["total_assets"]).all())
        self.assertTrue((c["debt_due_24m"] <= c["debt"] + 1).all())
        shares = [
            "floating_debt_share", "compute_cost_share", "compute_revenue_share", "backlog_cover",
            "gpu_asset_share", "liquidation_haircut", "funding_dependence", "top5_customer_share",
            "ig_customer_share", "circular_revenue_share",
        ]
        self.assertTrue(c[shares].apply(lambda s: s.between(0, 1)).all().all())
        self.assertTrue((c["equity_value_now"] <= c["equity_value_ref"] * 1.0001).all())


class BenchmarkRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.comp = {s: compare(cls.data, CFG, s) for s in SNAPSHOTS}

    def test_comparison_is_bounded_and_complete(self):
        for snap, comp in self.comp.items():
            for col in ("pd_1y", "pd_rn_1y", "pd_physical_1y_est"):
                self.assertTrue(comp[col].between(0, 1).all(), (snap, col))
            self.assertTrue((comp["pd_physical_1y_est"] < comp["pd_rn_1y"]).all())
            self.assertGreaterEqual(len(comp), 8, snap)

    def test_rank_stats_are_finite(self):
        s = rank_stats(self.comp["latest"])
        self.assertTrue(-1 <= s["spearman_model_vs_cds"] <= 1)
        self.assertTrue(0 < s["perm_pvalue"] <= 1)
        fit = rank_stats(self.comp["latest"], "approximate")
        self.assertLess(fit["n"], s["n"])

    def test_logodds_slope(self):
        from ai_pd.benchmark import logodds_slope
        from ai_pd.scorecard import inv_logit

        x = np.array([0.001, 0.01, 0.05, 0.2])
        self.assertAlmostEqual(logodds_slope(x, x), 1.0)
        doubled = inv_logit(2 * np.log(x / (1 - x)))
        self.assertAlmostEqual(logodds_slope(doubled, x), 2.0)
        self.assertTrue(np.isfinite(rank_stats(self.comp["latest"])["logodds_slope"]))

    def test_segment_gaps_cover_benchmarked_segments(self):
        g = segment_gaps(self.comp["latest"], CFG)
        self.assertEqual(set(g.index), set(self.comp["latest"]["segment"]))
        self.assertTrue(g["implied_prior_floor"].between(0, 1).all())

    def test_widest_credit_is_coreweave(self):
        # A sanity anchor rather than a model claim: the market's riskiest name.
        self.assertEqual(self.comp["latest"].iloc[0]["company_id"], "coreweave")

    def test_report_writes(self):
        with tempfile.TemporaryDirectory() as d:
            summary = write_report(self.data, CFG, Path(d))
            self.assertTrue((Path(d) / "benchmark.md").exists())
            self.assertEqual(set(summary), set(SNAPSHOTS))
            text = (Path(d) / "benchmark.md").read_text()
            self.assertIn("CoreWeave", text)

    def test_recovery_assumption_moves_market_pd_not_model_pd(self):
        lo = compare(self.data, CFG, "latest", recovery=0.2)
        hi = compare(self.data, CFG, "latest", recovery=0.6)
        np.testing.assert_allclose(lo["pd_1y"], hi["pd_1y"])
        self.assertTrue((hi["pd_rn_1y"] > lo["pd_rn_1y"]).all())


if __name__ == "__main__":
    unittest.main()
