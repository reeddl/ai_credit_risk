"""Step 5: real-company inputs from SEC filings, the boom panel and the weight fit."""

import io
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ai_pd.config import load_config
from ai_pd.features import RAW_COLUMNS, score_features
from ai_pd.fit import FITTED, auc, fit_logit, hand_weights, holdout
from ai_pd.panel import label, load_booms, to_raw
from ai_pd.panel import build as build_panel
from ai_pd.real import REAL, compare_to_benchmark, judgment_table, market_fields
from ai_pd.real import build as build_real
from ai_pd.xbrl import CompanyFacts, concept_whitelist, financial_inputs, shares_outstanding
from tests import sec_fixture as fx
from tools import fetch_sec

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
NOV = date(2026, 11, 15)


class TestXbrl(unittest.TestCase):
    cf = CompanyFacts(fx.companyfacts())

    def test_point_in_time_values(self):
        v = financial_inputs(self.cf, NOV, "example").values
        self.assertEqual(v["total_assets"], 2000)
        self.assertEqual(v["cash"], 400)  # cash + current investments
        self.assertEqual(v["debt"], 550)  # long-term debt + short-term borrowings
        self.assertEqual(v["debt_due_24m"], 150)  # 40 + 60 from the 10-K, + 50 short-term
        self.assertEqual(v["hard_assets"], 800)
        self.assertEqual(v["revenue_annual"], 625)  # FY25 460 + 9M26 495 - 9M25 330
        self.assertAlmostEqual(v["revenue_growth"], 180 / 120 - 1)
        self.assertAlmostEqual(v["gross_margin"], 0.6)
        self.assertAlmostEqual(v["monthly_net_burn"], -(120 - 80) / 12)
        self.assertTrue(math.isnan(v["employees"]))

    def test_nothing_filed_after_the_as_of_date_is_used(self):
        b = financial_inputs(self.cf, date(2026, 8, 10), "example")
        self.assertEqual(b.period_end, date(2026, 6, 30))
        self.assertEqual(b.values["total_assets"], 1900)
        self.assertEqual(b.values["revenue_annual"], 460 + 310 - 210)  # Q2 as first reported
        for s in b.sources:
            for d in str(s["date"] or "").split(" | "):
                if d:
                    self.assertLessEqual(date.fromisoformat(d), date(2026, 8, 10))

    def test_a_restatement_replaces_the_first_report(self):
        v = financial_inputs(self.cf, date(2026, 8, 25), "example").values
        self.assertEqual(v["revenue_annual"], 460 + 315 - 210)

    def test_every_value_has_a_source_row(self):
        b = financial_inputs(self.cf, NOV, "example")
        self.assertEqual({s["field"] for s in b.sources}, set(b.values))
        for s in b.sources:
            if s["kind"] in ("filing", "derived"):
                self.assertIn("SEC XBRL", s["source"])
                self.assertTrue(s["url"].startswith("https://www.sec.gov/Archives/edgar/data/9999901/"))

    def test_no_debt_tagged_means_no_debt(self):
        raw = fx.companyfacts()
        for k in ["LongTermDebt", "ShortTermBorrowings"]:
            raw["facts"]["us-gaap"].pop(k)
        b = financial_inputs(CompanyFacts(raw), NOV, "example")
        self.assertEqual(b.values["debt"], 0.0)
        kinds = {s["field"]: s["kind"] for s in b.sources}
        self.assertEqual(kinds["debt"], "assumed")

    def test_not_buildable_without_us_gaap_dollars(self):
        raw = {"cik": 1, "entityName": "IFRS Co", "facts": {"ifrs-full": {"Assets": {"units": {"EUR": []}}}}}
        b = financial_inputs(CompanyFacts(raw), NOV, "x")
        self.assertEqual(b.values, {})
        self.assertIn("no US GAAP facts in US dollars", b.problems)

    def test_shares(self):
        self.assertEqual(shares_outstanding(self.cf, NOV)[0], 1000.0)
        self.assertIsNone(shares_outstanding(self.cf, date(2026, 6, 1)))


class TestMissingInputsScoreNeutral(unittest.TestCase):
    def test_missing_employees_and_ppe_score_3(self):
        df = pd.read_csv(ROOT / "benchmark" / "data" / "companies_real.csv").head(2).copy()
        df.loc[0, ["employees", "hard_assets", "monthly_net_burn"]] = np.nan
        df.loc[0, "revenue_growth"] = np.nan
        s = score_features(df, CFG)
        for feat in ["labor_intensity", "collateral", "runway", "revenue"]:
            self.assertEqual(s.loc[0, feat], 3.0, feat)

    def test_a_known_zero_denominator_still_scores_as_before(self):
        df = pd.read_csv(ROOT / "benchmark" / "data" / "companies_real.csv").head(1).copy()
        df["monthly_net_burn"] = -1.0  # cash generative: unlimited runway, safest score
        self.assertEqual(score_features(df, CFG).loc[0, "runway"], 1.0)


class TestRealBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(1)
        path = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
        self.pull = fx.write_pull(Path(self.tmp.name), prices=fx.closes("2025-08-01", 400, path))
        self.universe = pd.DataFrame([
            {"company_id": "example", "ticker": "EXC", "name": "Example", "segment": "chip_designer",
             "fit": "good", "build": "xbrl", "reason": "test"},
            {"company_id": "tsmc", "ticker": "TSM", "name": "TSMC", "segment": "fab_equipment",
             "fit": "good", "build": "hand", "reason": "test"}])

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_every_raw_column(self):
        companies, sources, report = build_real(self.pull, NOV, self.universe)
        self.assertEqual(list(companies.columns), RAW_COLUMNS)
        ex = companies.set_index("company_id").loc["example"]
        self.assertEqual(ex.revenue_annual, 625)
        self.assertGreater(ex.equity_value_now, 0)
        self.assertGreaterEqual(ex.equity_value_ref, ex.equity_value_now)
        self.assertTrue(0.1 < ex.equity_vol < 0.6)
        defaults = sources[(sources.company_id == "example") & (sources.kind == "segment_default")]
        self.assertIn("moat_score", set(defaults.field))
        self.assertEqual(set(report.status), {"built", "copied from benchmark"})

    def test_hand_built_names_are_copied_unchanged(self):
        companies, _, _ = build_real(self.pull, NOV, self.universe)
        bench = pd.read_csv(ROOT / "benchmark" / "data" / "companies_real.csv").set_index("company_id")
        got = companies.set_index("company_id").loc["tsmc"]
        for c in RAW_COLUMNS[1:]:
            self.assertEqual(str(got[c]), str(bench.at["tsmc", c]), c)

    def test_hand_judgments_win_over_segment_defaults(self):
        j = judgment_table()
        self.assertIn(("nvidia", "moat_score"), j)

    def test_market_fields_need_a_year_of_prices(self):
        self.assertTrue(math.isnan(market_fields(None, 100.0, NOV)["equity_vol"]))


class TestUniverseFiles(unittest.TestCase):
    def test_universe(self):
        u = pd.read_csv(REAL / "universe.csv")
        self.assertFalse(u.company_id.duplicated().any())
        self.assertGreaterEqual(len(u), 60)
        self.assertTrue(u.segment.isin(CFG.segments).all())
        self.assertTrue(u.build.isin(["xbrl", "hand"]).all())
        self.assertEqual(set(u.loc[u.build == "hand", "company_id"]), {"tsmc", "sk_hynix", "softbank"})
        bench = set(pd.read_csv(ROOT / "benchmark" / "data" / "companies_real.csv").company_id)
        self.assertTrue(bench <= set(u.company_id))

    def test_segment_defaults_cover_every_segment(self):
        d = pd.read_csv(REAL / "segment_defaults.csv")
        self.assertEqual(set(d.segment), set(CFG.segments))

    def test_panel_firms(self):
        p = pd.read_csv(ROOT / "data" / "analogs" / "panel_firms.csv")
        self.assertFalse(p.firm_id.duplicated().any())
        self.assertTrue(p.boom.isin(load_booms().index).all())


class TestPanel(unittest.TestCase):
    def subs(self, extra=()):
        s = fx.submissions()
        s["filings"] += list(extra)
        return s

    def test_bankruptcy_8k_within_a_year_is_a_default(self):
        s = self.subs([{"form": "8-K", "filingDate": "2026-12-01", "items": "1.03,7.01"}])
        lab = label(date(2026, 2, 20), s, None)
        self.assertTrue(lab["default"] and lab["bankruptcy"])
        self.assertFalse(label(date(2025, 2, 20), s, None)["bankruptcy"])  # more than a year later

    def test_a_90_percent_price_fall_is_a_default(self):
        path = np.r_[np.full(50, 100.0), np.linspace(100, 9, 100), np.full(150, 9.0)]
        c = fx.closes("2025-01-01", 300, path)
        closes = pd.Series(c.close.values, index=pd.to_datetime(c.date))
        lab = label(date(2025, 3, 1), self.subs(), closes)
        self.assertTrue(lab["deep_loss"] and lab["default"])
        self.assertAlmostEqual(lab["max_drawdown_12m"], -0.91, places=2)

    def test_a_firm_that_goes_quiet_is_not_followed(self):
        s = {"filings": [{"form": "10-K", "filingDate": "2025-02-20", "items": ""}]}
        lab = label(date(2025, 2, 20), s, None)
        self.assertFalse(lab["default"])
        self.assertFalse(lab["followed"])

    def test_build_from_a_pull(self):
        with tempfile.TemporaryDirectory() as t:
            subs = self.subs([{"form": "8-K", "filingDate": "2026-09-01", "items": "1.03"}])
            pull = fx.write_pull(Path(t), who="analog", cid="example", subs=subs)
            firms = pd.DataFrame([{"firm_id": "example", "name": "Example", "ticker_then": "EXC",
                                   "boom": "test", "expected": "failed", "note": ""}])
            booms = pd.DataFrame({"start": [pd.Timestamp("2025-07-01")], "end": [pd.Timestamp("2027-06-30")]},
                                 index=pd.Index(["test"], name="boom"))
            panel, report = build_panel(pull, firms, booms)
        self.assertEqual(len(panel), 1)  # the FY2025 10-K filed 2026-02-20
        self.assertTrue(panel.iloc[0]["default"])
        self.assertEqual(report.iloc[0]["bankruptcy_8k"], "2026-09-01")
        raw = to_raw(panel)
        self.assertEqual(list(raw.columns), RAW_COLUMNS)
        self.assertEqual(len(score_features(raw, CFG)), 1)


def simulated(n, w_true, alphas, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.integers(-2, 3, size=(n, len(FITTED))).astype(float), columns=FITTED)
    g = np.array(list(alphas))[rng.integers(0, len(alphas), n)]
    lin = np.array([alphas[b] for b in g]) + X.values @ np.asarray(w_true)
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    return X, y, g


class TestFit(unittest.TestCase):
    w0 = hand_weights(CFG)
    alphas = {"a": -2.5, "b": -2.0, "c": -3.0, "d": -2.2}

    def test_recovers_weights_with_plenty_of_data(self):
        truth = self.w0 + np.array([0.3, -0.1, 0.2, 0.0, 0.25, -0.05])
        X, y, g = simulated(40000, truth, self.alphas)
        f = fit_logit(X, y, g, self.w0, tau=1.0)
        self.assertTrue(f.converged)
        np.testing.assert_allclose(f.weights.values, truth.values, atol=0.06)

    def test_stays_near_the_hand_weights_with_little_data(self):
        truth = self.w0 + 0.8
        X, y, g = simulated(150, truth, self.alphas, seed=3)
        f = fit_logit(X, y, g, self.w0)
        self.assertLess(np.abs(f.weights - self.w0).max(), 0.3)

    def test_holdout_reports_both_weight_sets_per_boom(self):
        X, y, g = simulated(2000, self.w0 * 3, self.alphas, seed=5)
        h = holdout(X, y, g, self.w0)
        self.assertEqual(len(h), 8)
        self.assertEqual(set(h.weights), {"fitted", "hand"})
        self.assertTrue(h.auc.between(0, 1).all())

    def test_auc(self):
        self.assertEqual(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(auc(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)


class TestFetchScript(unittest.TestCase):
    def test_refuses_without_a_user_agent(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(fetch_sec.main(["--out", "unused"]), 2)
            self.assertEqual(fetch_sec.main(["--user-agent", "nobody", "--out", "unused"]), 2)

    def test_dry_run_lists_sec_urls_only_for_named_firms(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(fetch_sec.main(["--dry-run", "--only", "nvidia", "riot"]), 0)
        out = buf.getvalue()
        self.assertIn("https://www.sec.gov/files/company_tickers.json", out)
        self.assertIn("universe:nvidia", out)
        self.assertIn("analog:riot", out)
        self.assertNotIn("universe:amd", out)

    def test_concept_whitelist_matches_the_builder(self):
        self.assertEqual(fetch_sec.load_concepts(), concept_whitelist())

    def test_reduce_drops_unlisted_concepts(self):
        r = fetch_sec.reduce_facts(fx.companyfacts(), concept_whitelist())
        self.assertNotIn("AccountsPayableCurrent", r["facts"]["us-gaap"])
        self.assertIn("Assets", r["facts"]["us-gaap"])
        self.assertEqual(CompanyFacts(r).cik, fx.CIK)


class TestAgainstHandBuiltRows(unittest.TestCase):
    """Built rows against the 12 hand-built benchmark rows. Runs once a pull has been built."""

    TOL = {"total_assets": 0.05, "revenue_annual": 0.15, "cash": 0.30, "debt": 0.30, "hard_assets": 0.20}

    def test_within_tolerance(self):
        path = REAL / "companies.csv"
        if not path.exists():
            self.skipTest("no SEC pull built yet (data/real/companies.csv)")
        check = compare_to_benchmark(pd.read_csv(path))
        self.assertEqual(check.company_id.nunique(), 12)
        for r in check[check.field.isin(self.TOL)].itertuples():
            if r.benchmark == 0 and r.built == 0:
                continue
            self.assertLessEqual(abs(r.ratio - 1), self.TOL[r.field], f"{r.company_id} {r.field}")


if __name__ == "__main__":
    unittest.main()
