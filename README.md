# AI company default model

A transparent probability-of-default model for AI companies, from power
suppliers and chip makers to model labs and app makers. There is no history
of AI companies defaulting through a long downturn, so the model is built
in layers that can each be checked, not fitted to defaults.

Version 1 runs end to end on synthetic companies. Every prior, weight and
shock size in `config/model.toml` is a placeholder.

## What counts as default

The first of these events within the horizon:

- a missed debt payment
- a bankruptcy filing
- a shutdown
- a sale or a down round that marks equity down 90% or more from the last
  price investors paid

This is wider than a classic credit PD. Most AI companies are funded with
equity, so shutdowns and deep markdowns are how most of their failures show up.

## How it works

1. **Segment prior.** Each of 13 segments starts from a 1-year PD.
2. **Scorecard.** Fourteen features are scored 1 (safest) to 5 (riskiest) and
   shift the PD on the log-odds scale. Compute sensitivity weighs most.
3. **Merton cross-check.** Where an equity value exists, a structural PD with
   the wider barrier (debt, or a 90% fall in equity) is computed. Large gaps
   from the scorecard are flagged, not averaged.
4. **Scenarios.** Compute glut, demand stall, funding freeze, rates up, and
   all four combined. Each moves raw inputs, re-scores the features and adds a
   segment-level shift. Compute glut and demand stall run off the shared
   compute demand driver (`config/driver.toml`): a cut to next year's big four
   hyperscaler AI capex estimate, with neocloud, lab and sovereign spending
   cut 1.5x as hard. Their shock sizes in `config/model.toml` apply at the
   reference cut (20%) and scale with the cut elsewhere.
5. **Calibration.** PDs map to implied rating grades (placeholder boundaries).
6. **Sensitivity.** Feature weights and segment priors are shifted one at a
   time to show which assumptions move PDs most.

```latex
\operatorname{logit}(PD_i) = \operatorname{logit}(PD_{seg(i)}) + \lambda_{seg(i)} \sum_k w_k (score_{ik} - 3) + \gamma_{seg(i)} \cdot s
```

`\lambda` is a per-segment weight scale (1 by default; 0.45 proposed for chip
designers and fabs, see section 6.2 of the model document).

## Run it

Needs Python 3.11+, numpy and pandas.

```bash
python -m ai_pd.run --n-per-segment 5 --seed 7 --out output
python -m ai_pd.run --compute-cut bust --out output_bust   # or mild, stall, or a fraction like 0.15
python -m unittest discover -s tests -t .
```

To score real companies, pass a CSV with the columns in
`ai_pd/features.py` (`RAW_COLUMNS`): `python -m ai_pd.run --companies my.csv`.

Outputs in `output/`: `companies.csv` (base, Merton and stressed PDs per
company), `feature_scores.csv`, `sensitivity.csv` and `summary.md`.

## Equity drawdown engine

`ai_eq` answers a different question from the credit model: if the market
cuts its estimate of AI compute demand by X%, how far does a stock fall?
It runs off the same driver. Per stock, the EPS elasticity is revenue
elasticity to the driver times operating leverage; a reverse DCF reads the
growth today's price implies after next year; a cut lowers next year's EPS
by `1 - elasticity * cut` (floored at 5% of consensus) and that growth by
`kappa` times the EPS cut, but never below the growth a 5.6x forward P/E
implies. Settings and placeholders are in `config/equity.toml`.

`kappa` and the floor are fitted on past drawdowns in `data/analogs/`
(`python -m ai_eq.calibrate`, report in `data/analogs/fit.md`). Each figure
there is marked sourced, derived or approximate.

```bash
python -m ai_eq.run --stocks my_stocks.csv --cut mild --cut bust --out output_eq
```

The stock CSV needs `name, segment, price, eps_next, gross_margin,
operating_margin`; see `ai_eq/inputs.py` for optional columns. Single-company
runs keep their input files outside the repo.

## Benchmark against CDS

`python -m ai_pd.benchmark` compares model PDs for 12 real companies with
PDs implied by their 5-year credit default swap spreads. Inputs come from
the latest public filings, with a source row per figure in
`benchmark/data/companies_sources.csv`; to rebuild them run
`python benchmark/build_inputs.py`. Results land in `benchmark/results/`
(the committed copy is from November 16, 2026). `python
benchmark/propose_weights.py` refits the chip and fab weight scale and writes
`benchmark/results/weight_proposal.md`. See section 6 of
`docs/model_document.md` for findings and caveats.

## Layout

| Path | What it holds |
| --- | --- |
| `config/model.toml` | Default definition, segments, features, scenarios, grade map |
| `config/driver.toml` | Compute demand driver: consensus capex, second-leg beta, named cuts, reference cut |
| `ai_risk/driver.py` | The driver, shared by the credit scenarios and the equity engine |
| `config/equity.toml` | Equity engine: valuation settings, k, margin hit, revenue elasticity and driver leg per segment |
| `ai_eq/exposure.py` | EPS elasticity to the driver |
| `ai_eq/rdcf.py` | Reverse DCF: implied growth after next year, and scenario value |
| `ai_eq/revision.py` | Growth revision: kappa and the floor multiple, and their fit on analog episodes |
| `ai_eq/calibrate.py` | Runs the fit and writes `data/analogs/fit.md` |
| `data/analogs/` | Analog drawdown episodes, a source row per figure, and the fit report |
| `ai_eq/drawdown.py` | Low, central and high drawdown per stock and cut; headline move per 10% cut |
| `ai_pd/defaults.py` | Default events and markdown labeling |
| `ai_pd/features.py` | Raw inputs to metrics to 1-5 scores |
| `ai_pd/scorecard.py` | Log-odds combination, 1-year and 3-year PD |
| `ai_pd/merton.py` | Structural cross-check with the wider barrier |
| `ai_pd/scenarios.py` | Stress scenarios |
| `ai_pd/synthetic.py` | Synthetic companies per segment |
| `ai_pd/sensitivity.py` | One-at-a-time sensitivity |
| `ai_pd/market_implied.py` | CDS spread to PD, risk-neutral to physical, rank statistics |
| `ai_pd/benchmark.py` | Model vs CDS comparison and report |
| `benchmark/data/` | CDS quotes, filing figures, judgments, segment map, built inputs |
| `benchmark/build_inputs.py` | Filings and judgments to model inputs, with provenance |
| `benchmark/results/` | Latest benchmark report and tables |
| `docs/model_document.md` | Draft model document |
