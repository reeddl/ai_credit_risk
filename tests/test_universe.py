"""The eight-name equity universe (data/universe/) and its credit tail check."""

import math
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from ai_eq.config import CASES
from ai_eq.universe import (SNAPSHOT_DATE, UNIVERSE, headline_markdown, load_credit_inputs,
                            load_stocks, run)
from ai_pd.features import RAW_COLUMNS
from ai_risk.driver import load_driver
from benchmark.build_inputs import DATA as BENCHMARK_DATA
from benchmark.build_inputs import build

STOCKS = load_stocks()
SOURCES = pd.read_csv(UNIVERSE / "stock_sources.csv", dtype={"value": str})
CREDIT_SOURCES = pd.read_csv(UNIVERSE / "credit_sources.csv")
HEAD, TAIL = run()
SNAP = date.fromisoformat(SNAPSHOT_DATE)
FIGURES = ["price", "eps_next", "gross_margin", "operating_margin", "years_to_first_flow",
           "target_year", "fiscal_year_end"] + [f"revenue_elasticity_{c}" for c in CASES]
NAMES = {"nvidia", "amd", "broadcom", "tsmc", "sk_hynix", "oracle", "coreweave", "vistra"}


class TestUniverseData(unittest.TestCase):
    def test_the_eight_names(self):
        self.assertEqual(set(STOCKS.company_id), NAMES)
        self.assertFalse(STOCKS.company_id.duplicated().any())

    def test_every_figure_has_one_matching_source_row(self):
        src = SOURCES.set_index(["company_id", "field"])
        self.assertFalse(src.index.duplicated().any())
        for row in STOCKS.itertuples():
            for f in FIGURES:
                v = getattr(row, f)
                if isinstance(v, float) and math.isnan(v):
                    self.assertNotIn((row.company_id, f), src.index)
                    continue
                self.assertIn((row.company_id, f), src.index, f"{row.company_id} {f}")
                s = src.loc[(row.company_id, f), "value"]
                if isinstance(v, str):
                    self.assertEqual(s, v)
                else:
                    self.assertAlmostEqual(float(s), float(v), places=9, msg=f"{row.company_id} {f}")

    def test_source_rows_are_complete(self):
        self.assertTrue(SOURCES.status.isin(["sourced", "derived", "approximate", "judgment"]).all())
        self.assertTrue(SOURCES.source.notna().all())
        self.assertTrue(SOURCES.note.notna().all())
        sourced = SOURCES[SOURCES.status == "sourced"]
        self.assertTrue(sourced.url.str.startswith("https://").all())

    def test_no_future_dates(self):
        # Judgment rows in the credit sources carry no date, as in the benchmark set.
        undated = CREDIT_SOURCES[CREDIT_SOURCES.date.isna()]
        self.assertTrue((undated.kind == "judgment").all())
        for d in pd.concat([SOURCES.as_of, CREDIT_SOURCES.date.dropna(), STOCKS.price_date]):
            for part in str(d).split(" | "):
                self.assertLessEqual(date.fromisoformat(part), SNAP, part)

    def test_consensus_figures_are_flagged_as_free_snapshots(self):
        eps = SOURCES[SOURCES.field == "eps_next"]
        self.assertEqual(len(eps), len(STOCKS))
        self.assertTrue(eps.free_snapshot.all())
        consensus = SOURCES[SOURCES.source.str.contains("estimate")]
        self.assertTrue(consensus.free_snapshot.all())
        self.assertFalse(SOURCES[SOURCES.field == "price"].free_snapshot.any())

    def test_target_year_and_years_to_first_flow(self):
        for row in STOCKS.itertuples():
            px, fye = date.fromisoformat(row.price_date), date.fromisoformat(row.fiscal_year_end)
            months = (fye - px).days / 30.44
            self.assertTrue(9 <= months <= 21, row.company_id)
            self.assertAlmostEqual(row.years_to_first_flow, (fye - px).days / 365.25, places=3)

    def test_engine_runs_exactly_on_profitable_names(self):
        self.assertTrue(((STOCKS.eps_next > 0) == STOCKS.equity_engine).all())
        self.assertEqual(set(STOCKS.loc[~STOCKS.equity_engine, "company_id"]), {"coreweave"})

    def test_credit_inputs_cover_every_name(self):
        credit = load_credit_inputs(STOCKS.company_id)
        self.assertEqual(set(credit.company_id), NAMES)
        self.assertFalse(credit[RAW_COLUMNS].isna().any().any())

    def test_universe_credit_inputs_have_a_source_row_each(self):
        credit = pd.read_csv(UNIVERSE / "credit_inputs.csv")
        for cid in credit.company_id:
            fields = set(CREDIT_SOURCES.loc[CREDIT_SOURCES.company_id == cid, "field"])
            self.assertEqual(fields, set(RAW_COLUMNS) - {"company_id", "segment", "public"}, cid)

    def test_builder_reproduces_committed_credit_inputs(self):
        companies, prov = build(UNIVERSE)
        pd.testing.assert_frame_equal(companies.reset_index(drop=True),
                                      pd.read_csv(UNIVERSE / "credit_inputs.csv"), check_dtype=False)
        self.assertEqual(len(prov), len(CREDIT_SOURCES))

    def test_benchmark_build_is_unchanged(self):
        companies, _ = build()
        pd.testing.assert_frame_equal(companies.reset_index(drop=True),
                                      pd.read_csv(BENCHMARK_DATA / "companies_real.csv"), check_dtype=False)


class TestTailCheck(unittest.TestCase):
    def test_runs_for_every_name_and_cut(self):
        cuts = set(load_driver().cuts)
        self.assertEqual(len(TAIL), len(STOCKS) * len(cuts))
        for cid in NAMES:
            rows = TAIL[TAIL.company_id == cid]
            self.assertEqual(set(rows.scenario), cuts, cid)
            for col in ["credit_pd_base", "credit_pd_stressed", "merton_pd_base"]:
                self.assertTrue(rows[col].between(0, 1).all(), f"{cid} {col}")

    def test_equity_columns_where_the_engine_runs(self):
        eq = TAIL[TAIL.company_id != "coreweave"]
        for col in ["drawdown_low", "drawdown_central", "drawdown_high",
                    "merton_pd_after_central", "merton_pd_after_high", "equity_after_high_vs_ref"]:
            self.assertFalse(eq[col].isna().any(), col)
        self.assertTrue((eq.drawdown_low >= eq.drawdown_central).all())
        self.assertTrue((eq.drawdown_central >= eq.drawdown_high).all())
        cw = TAIL[TAIL.company_id == "coreweave"]
        self.assertTrue(cw.drawdown_central.isna().all())
        self.assertTrue((cw["flags"] == "credit_only").all())

    def test_stressed_pd_is_the_worse_driver_scenario(self):
        self.assertTrue((TAIL.credit_pd_stressed >= TAIL.credit_pd_base - 1e-12).all())
        self.assertTrue(TAIL.credit_stress_scenario.isin(["demand_stall", "compute_glut"]).all())

    def test_a_deeper_drawdown_never_lowers_merton(self):
        eq = TAIL[TAIL.company_id != "coreweave"]
        self.assertTrue((eq.merton_pd_after_high >= eq.merton_pd_after_central - 1e-12).all())
        self.assertTrue((eq.merton_pd_after_central >= eq.merton_pd_base - 1e-12).all())

    def test_crosses_90_matches_its_definition(self):
        eq = TAIL[TAIL.company_id != "coreweave"]
        crosses = eq["flags"].fillna("").str.contains("crosses_90")
        self.assertTrue((crosses == (eq.equity_after_high_vs_ref <= 0.1)).all())
        self.assertTrue((eq.loc[crosses, "merton_pd_after_high"] == 1.0).all())

    def test_deeper_cuts_hurt_more(self):
        for cid, rows in TAIL.groupby("company_id"):
            rows = rows.sort_values("cut")
            self.assertTrue(rows.credit_pd_stressed.is_monotonic_increasing, cid)
            if cid != "coreweave":
                self.assertTrue(rows.drawdown_central.is_monotonic_decreasing, cid)


class TestHeadline(unittest.TestCase):
    def test_ranks_all_eight_most_exposed_first(self):
        self.assertEqual(len(HEAD), 8)
        self.assertEqual(list(HEAD["rank"]), list(range(1, 9)))
        moves = HEAD.move_10pct_central.dropna()
        self.assertTrue(moves.is_monotonic_increasing)
        self.assertEqual(HEAD.iloc[-1]["name"], "CoreWeave")

    def test_markdown_has_a_row_per_name(self):
        md = headline_markdown(HEAD)
        for name in HEAD.name:
            self.assertIn(f"| {name} |", md)
        self.assertIn("n/a (loss-making)", md)

    def test_committed_results_reproduce(self):
        out = UNIVERSE / "results"
        committed = pd.read_csv(out / "tail_check.csv")
        pd.testing.assert_frame_equal(committed.drop(columns="flags"), TAIL.drop(columns="flags"),
                                      check_dtype=False, rtol=1e-5)
        self.assertEqual(list(committed["flags"].fillna("")), list(TAIL["flags"].fillna("")))
        self.assertEqual((out / "headline.md").read_text(), headline_markdown(HEAD))


if __name__ == "__main__":
    unittest.main()
