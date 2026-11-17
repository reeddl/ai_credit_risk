"""The eight-name universe: equity drawdowns joined to the credit model's tail.

Inputs live in ``data/universe/``:

- ``stocks.csv``: one row per name with the equity engine's inputs. Names
  whose consensus EPS is negative (CoreWeave) carry ``equity_engine = False``
  and run in the credit model only.
- ``stock_sources.csv``: one row per figure in ``stocks.csv`` with its status
  (sourced, derived, approximate or judgment), source, URL and as-of date.
  Consensus figures are free snapshots and say so in ``free_snapshot``.
- ``credit_inputs.csv``: credit model inputs for names not already in the
  CDS benchmark set (built by ``python benchmark/build_inputs.py --universe``).
  The other names' credit inputs come from ``benchmark/data/companies_real.csv``.

Tail check, per name and named cut (mild, stall, bust):

- the equity engine's low, central and high drawdown;
- the credit model's stressed 1-year PD: the worse of its demand stall and
  compute glut scenarios at that cut (both run off the same driver);
- the Merton PD (90% markdown barrier) with equity moved by the central and
  the high drawdown, everything else as in the credit inputs.

Two flags. ``tails_disagree``: the stressed scorecard PD and the Merton PD
after the high drawdown differ by more than ``merton_gap_flag_logodds`` on
the log-odds scale, with both PDs floored at ``PD_FLOOR`` first so that two
negligible PDs never count as a disagreement. ``crosses_90``: the high
drawdown alone takes equity to 10% or less of the reference value (the
52-week high), which is a default event under the credit model's definition.

Run: ``python -m ai_eq.universe --out data/universe/results``
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from ai_eq.config import CASES, EquityConfig, load_equity_config
from ai_eq.drawdown import HEADLINE_CUT, drawdown
from ai_eq.exposure import eps_elasticity
from ai_eq.inputs import stocks_from_frame
from ai_eq.rdcf import implied_growth
from ai_eq.drawdown import Valuation
from ai_pd.config import ModelConfig, load_config
from ai_pd.merton import merton_pd
from ai_pd.pipeline import run_model
from ai_risk.driver import Driver, load_driver

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "universe"
BENCHMARK_INPUTS = ROOT / "benchmark" / "data" / "companies_real.csv"

# Latest date any figure in the universe may carry.
SNAPSHOT_DATE = "2026-11-17"

# PDs are floored here before the log-odds comparison (the Basel minimum,
# 3 basis points), so that two negligible PDs never count as disagreeing.
PD_FLOOR = 0.0003

CREDIT_STRESS = ("demand_stall", "compute_glut")


def load_stocks(data: Path = UNIVERSE) -> pd.DataFrame:
    df = pd.read_csv(data / "stocks.csv")
    df["equity_engine"] = df["equity_engine"].astype(str).str.lower() == "true"
    return df


def load_credit_inputs(ids, data: Path = UNIVERSE, benchmark: Path = BENCHMARK_INPUTS) -> pd.DataFrame:
    """Credit inputs for ``ids``: the benchmark set first, then the universe's own file."""
    frames = [pd.read_csv(benchmark)]
    extra = data / "credit_inputs.csv"
    if extra.exists():
        frames.append(pd.read_csv(extra))
    df = pd.concat(frames, ignore_index=True)
    dup = df["company_id"].duplicated()
    if dup.any():
        raise ValueError(f"credit inputs listed twice: {sorted(df.loc[dup, 'company_id'])}")
    missing = sorted(set(ids) - set(df["company_id"]))
    if missing:
        raise ValueError(f"no credit inputs for: {missing}")
    return df.set_index("company_id").loc[list(ids)].reset_index()


def _logit(p: float) -> float:
    p = min(max(p, PD_FLOOR), 1 - 1e-9)
    return math.log(p / (1 - p))


def tail_check(stocks: pd.DataFrame, credit: pd.DataFrame, eq_cfg: EquityConfig | None = None,
               pd_cfg: ModelConfig | None = None, driver: Driver | None = None,
               cuts=None) -> pd.DataFrame:
    """One row per name and cut. Every name gets credit columns; equity columns only where the engine runs."""
    eq_cfg = eq_cfg or load_equity_config()
    pd_cfg = pd_cfg or load_config()
    driver = driver or load_driver()
    cuts = list(driver.cuts) if cuts is None else list(cuts)
    r = pd_cfg.model["risk_free_rate"]
    threshold = pd_cfg.default_definition["markdown_threshold"]
    gap_flag = pd_cfg.model["merton_gap_flag_logodds"]

    eq = {row.company_id: s for row, s in zip(
        stocks[stocks.equity_engine].itertuples(), stocks_from_frame(stocks[stocks.equity_engine]))}
    credit = credit.set_index("company_id")
    rows = []
    for cut in cuts:
        res = run_model(credit.reset_index(), pd_cfg, cut, driver).companies.set_index("company_id")
        for row in stocks.itertuples():
            cid = row.company_id
            c = res.loc[cid]
            inp = credit.loc[cid]
            worst = max(CREDIT_STRESS, key=lambda s: c[f"pd_1y_{s}"])
            out = {"company_id": cid, "name": row.name, "scenario": cut, "cut": driver.resolve(cut),
                   "credit_pd_base": c["pd_1y"], "credit_pd_stressed": c[f"pd_1y_{worst}"],
                   "credit_stress_scenario": worst, "merton_pd_base": c["pd_merton_1y"]}
            flags = []
            if cid in eq:
                s = eq[cid]
                for case in CASES:
                    out[f"drawdown_{case}"] = drawdown(s, cut, case, eq_cfg, driver)
                for case in ("central", "high"):
                    out[f"merton_pd_after_{case}"] = merton_pd(
                        inp.equity_value_now * (1 + out[f"drawdown_{case}"]), inp.equity_value_ref,
                        inp.debt, inp.equity_vol, r=r, threshold=threshold)
                out["equity_after_high_vs_ref"] = (
                    inp.equity_value_now * (1 + out["drawdown_high"]) / inp.equity_value_ref)
                gap = _logit(out["merton_pd_after_high"]) - _logit(out["credit_pd_stressed"])
                out["tail_gap_logodds"] = gap
                if abs(gap) > gap_flag:
                    flags.append("tails_disagree")
                if out["equity_after_high_vs_ref"] <= 1 - threshold:
                    flags.append("crosses_90")
            else:
                flags.append("credit_only")
            out["flags"] = " ".join(flags)
            rows.append(out)
    cols = ["company_id", "name", "scenario", "cut", "drawdown_low", "drawdown_central", "drawdown_high",
            "credit_pd_base", "credit_pd_stressed", "credit_stress_scenario", "merton_pd_base",
            "merton_pd_after_central", "merton_pd_after_high", "equity_after_high_vs_ref",
            "tail_gap_logodds", "flags"]
    return pd.DataFrame(rows).reindex(columns=cols)


def headline_table(stocks: pd.DataFrame, tail: pd.DataFrame, eq_cfg: EquityConfig | None = None,
                   driver: Driver | None = None) -> pd.DataFrame:
    """All names ranked by central price move per 10% cut, most exposed first; credit-only names last."""
    eq_cfg = eq_cfg or load_equity_config()
    driver = driver or load_driver()
    eq = dict(zip(stocks.loc[stocks.equity_engine, "company_id"],
                  stocks_from_frame(stocks[stocks.equity_engine])))
    bust = tail[tail.scenario == "bust"].set_index("company_id")
    stall = tail[tail.scenario == "stall"].set_index("company_id")
    rows = []
    for row in stocks.itertuples():
        cid = row.company_id
        out = {"name": row.name, "segment": row.segment, "driver_leg": eq_cfg.leg(row.segment)}
        if cid in eq:
            s = eq[cid]
            val = Valuation.for_stock(s, eq_cfg)
            out["forward_pe"] = s.price / s.eps_next
            out["implied_growth"] = implied_growth(s.price, s.eps_next, val.r, **val.kw)
            out["eps_elasticity_central"] = eps_elasticity(s, eq_cfg)["central"]
            for case in CASES:
                out[f"move_10pct_{case}"] = drawdown(s, HEADLINE_CUT, case, eq_cfg, driver)
            out["stall_central"] = stall.loc[cid, "drawdown_central"]
            out["bust_central"] = bust.loc[cid, "drawdown_central"]
            out["bust_high"] = bust.loc[cid, "drawdown_high"]
        out["credit_pd_base"] = bust.loc[cid, "credit_pd_base"]
        out["credit_pd_bust"] = bust.loc[cid, "credit_pd_stressed"]
        out["merton_pd_bust_high"] = bust.loc[cid, "merton_pd_after_high"]
        out["bust_flags"] = bust.loc[cid, "flags"]
        rows.append(out)
    df = pd.DataFrame(rows)
    df = df.sort_values("move_10pct_central", na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def _pct(x, signed=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.0%}" if signed else f"{x:.1%}"


def headline_markdown(head: pd.DataFrame) -> str:
    lines = [
        "# Headline: price move per 10% cut to the compute demand estimate",
        "",
        f"Inputs as of {SNAPSHOT_DATE} (prices are 2026-11-13 closes). Central case, ranked most exposed first.",
        "Low and high are the engine's band. Credit PDs are 1-year, with default including a 90% markdown.",
        "",
        "| # | Name | Leg | Fwd P/E | Per 10% cut (low / central / high) | Stall | Bust | Bust, high case "
        "| Credit PD base | Credit PD bust | Merton PD after bust high | Flags at bust |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in head.itertuples():
        if isinstance(r.move_10pct_central, float) and math.isnan(r.move_10pct_central):
            band = "n/a (loss-making)"
            pe = "n/a"
        else:
            band = f"{_pct(r.move_10pct_low)} / **{_pct(r.move_10pct_central)}** / {_pct(r.move_10pct_high)}"
            pe = f"{r.forward_pe:.1f}x"
        lines.append(
            f"| {r.rank} | {r.name} | {r.driver_leg} | {pe} | {band} | {_pct(r.stall_central)} | "
            f"{_pct(r.bust_central)} | {_pct(r.bust_high)} | {_pct(r.credit_pd_base, False)} | "
            f"{_pct(r.credit_pd_bust, False)} | {_pct(r.merton_pd_bust_high, False)} | {r.bust_flags or '-'} |")
    lines += [
        "",
        "Named cuts: mild 10%, stall 20%, bust 35% to the big four's 2027 AI capex estimate; names on the",
        "credit-funded leg (neocloud segment) take 1.5x the cut. Flags: `tails_disagree` means the credit",
        "model's stressed PD and the Merton PD after the high drawdown differ by more than 1.5 in log-odds;",
        "`crosses_90` means the high drawdown alone takes equity below 10% of its 52-week high;",
        "`credit_only` means consensus EPS is negative, so the equity engine does not run.",
        "",
    ]
    return "\n".join(lines)


def run(data: Path = UNIVERSE, out: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stocks = load_stocks(data)
    credit = load_credit_inputs(stocks["company_id"], data)
    eq_cfg, pd_cfg, driver = load_equity_config(), load_config(), load_driver()
    tail = tail_check(stocks, credit, eq_cfg, pd_cfg, driver)
    head = headline_table(stocks, tail, eq_cfg, driver)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        tail.to_csv(out / "tail_check.csv", index=False, float_format="%.6g")
        head.to_csv(out / "headline.csv", index=False, float_format="%.6g")
        (out / "headline.md").write_text(headline_markdown(head))
    return head, tail


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the equity universe and its credit tail check.")
    p.add_argument("--data", default=str(UNIVERSE))
    p.add_argument("--out", help="directory for headline.csv, headline.md and tail_check.csv")
    a = p.parse_args(argv)
    head, tail = run(Path(a.data), Path(a.out) if a.out else None)
    print(headline_markdown(head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
