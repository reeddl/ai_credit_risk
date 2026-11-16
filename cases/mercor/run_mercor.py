"""Run Mercor through the AI company default model.

    python cases/mercor/run_mercor.py

Reads cases/mercor/inputs.csv (one row per model input, with a base value,
a low and high value where the input is uncertain, and a source or reason).
Writes cases/mercor/results/: results.md plus CSVs for the base run,
feature drivers, input sensitivity and model sensitivity.

Nothing here changes the model. The "lose a top lab" what-if at the end is
specific to this case: the config's five scenarios have no customer-loss
shock, which is Mercor's largest risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai_pd.config import load_config  # noqa: E402
from ai_pd.features import RAW_COLUMNS, compute_metrics, score_features  # noqa: E402
from ai_pd.merton import merton_pd  # noqa: E402
from ai_pd.pipeline import run_model  # noqa: E402
from ai_pd.scorecard import feature_contribution, inv_logit, logit, score_pd  # noqa: E402
from ai_pd.sensitivity import sensitivity_table  # noqa: E402

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs.csv"
OUT = HERE / "results"
TEXT_FIELDS = {"company_id", "segment", "revenue_stage", "public"}

# "Lose a top lab" what-if: share of gross revenue lost, and the equity
# mark after it (judgment, not a forecast).
LOSS_SHARES = (0.25, 0.50)
LOSS_EQUITY_MARK = {0.25: 0.6, 0.50: 0.35}


def _value(field: str, raw: str):
    if field == "public":
        return str(raw).strip().lower() == "true"
    if field in TEXT_FIELDS:
        return str(raw)
    return float(raw)


def load_inputs(path: Path = INPUTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (one-row company frame, the long inputs table)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in RAW_COLUMNS if c not in set(table["field"])]
    if missing:
        raise ValueError(f"inputs.csv is missing fields: {missing}")
    row = {r.field: _value(r.field, r.base) for r in table.itertuples()}
    return pd.DataFrame([row])[RAW_COLUMNS], table


def base_pd(df: pd.DataFrame, cfg) -> float:
    scores = score_features(df, cfg)
    return float(score_pd(df["segment"], scores, cfg)["pd_1y"].iloc[0])


def drivers(df: pd.DataFrame, cfg) -> pd.DataFrame:
    metrics = compute_metrics(df)
    scores = score_features(df, cfg)
    contrib = feature_contribution(scores, cfg, df["segment"])
    rows = []
    for feat_id, feat in cfg.features.items():
        rows.append(
            {
                "feature": feat_id,
                "metric": feat["metric"],
                "value": metrics[feat["metric"]].iloc[0],
                "score": scores[feat_id].iloc[0],
                "weight": feat["weight"],
                "logodds": contrib[feat_id].iloc[0],
            }
        )
    out = pd.DataFrame(rows)
    return out.reindex(out["logodds"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def input_sensitivity(df: pd.DataFrame, table: pd.DataFrame, cfg) -> pd.DataFrame:
    """Move one input at a time to its low and high value."""
    base = base_pd(df, cfg)
    rows = []
    for r in table.itertuples():
        if r.field in ("company_id", "public") or (not r.low and not r.high):
            continue
        out = {"input": r.field, "base_value": r.base}
        for side in ("low", "high"):
            raw = getattr(r, side)
            if not raw:
                out[f"{side}_value"], out[f"pd_{side}"] = "", base
                continue
            shocked = df.copy()
            shocked[r.field] = _value(r.field, raw)
            out[f"{side}_value"], out[f"pd_{side}"] = raw, base_pd(shocked, cfg)
        rows.append(out)
    sens = pd.DataFrame(rows)
    sens["pd_base"] = base
    sens["swing"] = (sens["pd_high"] - sens["pd_low"]).abs()
    return sens.sort_values("swing", ascending=False, ignore_index=True)


def lose_a_lab(df: pd.DataFrame, cfg, share: float) -> dict:
    """Mercor loses `share` of gross revenue; operating costs stay put.

    Contractor pay is variable (paid per hour), so gross margin holds.
    Operating costs (staff, inference) are taken as gross profit less
    today's free cash flow, and do not shrink within the year.
    """
    s = df.copy()
    rev, gm, burn = (float(s[c].iloc[0]) for c in ("revenue_annual", "gross_margin", "monthly_net_burn"))
    opex = rev * gm + 12.0 * burn
    new_rev = rev * (1.0 - share)
    new_burn = (opex - new_rev * gm) / 12.0
    cash = float(s["cash"].iloc[0])
    s["revenue_annual"] = new_rev
    s["revenue_growth"] = -share
    s["monthly_net_burn"] = new_burn
    s["margin_path_score"] = 4.0
    s["funding_dependence"] = float(np.clip(12.0 * max(new_burn, 0.0) / cash - 1.0, 0.0, 1.0)) if cash > 0 else 1.0
    s["equity_value_now"] = s["equity_value_ref"] * LOSS_EQUITY_MARK[share]
    scores = score_features(s, cfg)
    sev = {"demand_stall": 1.0}
    pd_1y = float(score_pd(s["segment"], scores, cfg, sev)["pd_1y"].iloc[0])
    r = cfg.model["risk_free_rate"]
    thr = cfg.default_definition["markdown_threshold"]
    pm = merton_pd(
        float(s["equity_value_now"].iloc[0]), float(s["equity_value_ref"].iloc[0]),
        float(s["debt"].iloc[0]), float(s["equity_vol"].iloc[0]), r=r, threshold=thr,
    )
    return {
        "revenue_lost": share,
        "new_revenue": new_rev,
        "new_monthly_burn": new_burn,
        "runway_months": cash / new_burn if new_burn > 0 else float("inf"),
        "equity_mark": LOSS_EQUITY_MARK[share],
        "pd_1y": pd_1y,
        "pd_merton_1y": pm,
    }


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    cfg = load_config()
    df, table = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)

    results = run_model(df, cfg)
    c = results.companies.iloc[0]
    drv = drivers(df, cfg)
    sens = input_sensitivity(df, table, cfg)
    model_sens = sensitivity_table(df, cfg)

    alt_seg = table.set_index("field").loc["segment", "low"]
    df_alt = df.copy()
    df_alt["segment"] = alt_seg
    alt = run_model(df_alt, cfg).companies.iloc[0]

    losses = pd.DataFrame([lose_a_lab(df, cfg, s) for s in LOSS_SHARES])

    # The CDS benchmark found feature weights too strong for chip and fab
    # names (weight_scale 0.45). Show the same scale on this segment.
    cfg_scaled = cfg.copy()
    cfg_scaled.segments[df["segment"].iloc[0]]["weight_scale"] = 0.45
    pd_scaled = base_pd(df, cfg_scaled)

    # Merton: equity value at which the 90% markdown barrier bites.
    barrier = (1 - cfg.default_definition["markdown_threshold"]) * float(df["equity_value_ref"].iloc[0])
    vol_rows = []
    for vol in (0.6, 0.8, 1.0):
        vol_rows.append((vol, merton_pd(float(df["equity_value_now"].iloc[0]), float(df["equity_value_ref"].iloc[0]),
                                        0.0, vol, r=cfg.model["risk_free_rate"])))

    results.companies.to_csv(OUT / "base_run.csv", index=False)
    drv.to_csv(OUT / "drivers.csv", index=False)
    sens.to_csv(OUT / "input_sensitivity.csv", index=False)
    model_sens.to_csv(OUT / "model_sensitivity.csv", index=False)
    losses.to_csv(OUT / "lose_a_lab.csv", index=False)

    scen = [n for n in cfg.scenarios]
    seg_name = cfg.segments[c["segment"]]["name"]
    prior = cfg.segments[c["segment"]]["prior_pd_1y"]
    lines = [
        "# Mercor: model results",
        "",
        "Generated by `python cases/mercor/run_mercor.py` from `cases/mercor/inputs.csv`.",
        "Inputs are public figures as of November 16, 2026, plus stated estimates; see the inputs file for every source.",
        "Model priors, weights and shocks are the placeholders in `config/model.toml`.",
        "",
        "## Headline",
        "",
        f"- Segment: {seg_name} (prior 1-year PD {pct(prior)})",
        f"- 1-year PD: **{pct(c['pd_1y'])}** (implied grade {c['implied_grade']}); 3-year PD {pct(c['pd_3y'])}",
        f"- Log-odds: prior {c['logodds_prior']:.2f}, features {c['logodds_features']:+.2f}",
        f"- Merton cross-check (no debt, 90% markdown barrier at ${barrier/1e9:.0f}B equity): {pct(c['pd_merton_1y'])}"
        + (" (flagged: gap over 1.5 log-odds)" if c["merton_gap_flag"] else ""),
        f"- With the strong-moat segment instead: {pct(alt['pd_1y'])}",
        f"- With feature weights scaled by 0.45 (the chip/fab fit from the CDS benchmark): {pct(pd_scaled)}",
        "",
        "## Stress scenarios (config)",
        "",
        "| Scenario | Scorecard PD | Merton PD |",
        "| --- | --- | --- |",
        f"| Base | {pct(c['pd_1y'])} | {pct(c['pd_merton_1y'])} |",
    ]
    for n in scen:
        lines.append(f"| {n} | {pct(c[f'pd_1y_{n}'])} | {pct(c[f'pd_merton_1y_{n}'])} |")
    lines += [
        "",
        "## What-if: lose a top lab (case-specific, not in the config)",
        "",
        "Gross margin holds (contractors are paid per hour), operating costs do not shrink, demand-stall segment shift applied, equity re-marked.",
        "",
        "| Revenue lost | Monthly burn | Runway | Equity mark | Scorecard PD | Merton PD |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in losses.itertuples():
        runway = "no limit" if not np.isfinite(r.runway_months) else f"{r.runway_months:.0f} months"
        lines.append(
            f"| {r.revenue_lost:.0%} | ${r.new_monthly_burn/1e6:.0f}M | {runway} | {r.equity_mark:.0%} of $20B "
            f"| {pct(r.pd_1y)} | {pct(r.pd_merton_1y)} |"
        )
    lines += ["", "## Feature drivers (log-odds vs segment prior)", "", "| Feature | Value | Score | Log-odds |", "| --- | --- | --- | --- |"]
    for r in drv.itertuples():
        v = r.value
        val = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.2f}"
        lines.append(f"| {r.feature} | {val} | {r.score:.0f} | {r.logodds:+.2f} |")
    lines += ["", "## Input sensitivity (one at a time)", "", "| Input | Low | High | PD low | PD high |", "| --- | --- | --- | --- | --- |"]
    for r in sens.head(10).itertuples():
        lines.append(f"| {r.input} | {r.low_value} | {r.high_value} | {pct(r.pd_low)} | {pct(r.pd_high)} |")
    lines += ["", "## Merton volatility", "", "| Equity vol | Merton PD |", "| --- | --- |"]
    lines += [f"| {v:.1f} | {pct(p)} |" for v, p in vol_rows]
    lines.append("")
    (OUT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
