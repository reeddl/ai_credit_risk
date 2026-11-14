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
   segment-level shift.
5. **Calibration.** PDs map to implied rating grades (placeholder boundaries).
6. **Sensitivity.** Feature weights and segment priors are shifted one at a
   time to show which assumptions move PDs most.

```latex
\operatorname{logit}(PD_i) = \operatorname{logit}(PD_{seg(i)}) + \sum_k w_k (score_{ik} - 3) + \gamma_{seg(i)} \cdot s
```

## Run it

Needs Python 3.11+, numpy and pandas.

```bash
python -m ai_pd.run --n-per-segment 5 --seed 7 --out output
python -m unittest discover -s tests -t .
```

To score real companies, pass a CSV with the columns in
`ai_pd/features.py` (`RAW_COLUMNS`): `python -m ai_pd.run --companies my.csv`.

Outputs in `output/`: `companies.csv` (base, Merton and stressed PDs per
company), `feature_scores.csv`, `sensitivity.csv` and `summary.md`.

## Layout

| Path | What it holds |
| --- | --- |
| `config/model.toml` | Default definition, segments, features, scenarios, grade map |
| `ai_pd/defaults.py` | Default events and markdown labeling |
| `ai_pd/features.py` | Raw inputs to metrics to 1-5 scores |
| `ai_pd/scorecard.py` | Log-odds combination, 1-year and 3-year PD |
| `ai_pd/merton.py` | Structural cross-check with the wider barrier |
| `ai_pd/scenarios.py` | Stress scenarios |
| `ai_pd/synthetic.py` | Synthetic companies per segment |
| `ai_pd/sensitivity.py` | One-at-a-time sensitivity |
| `docs/model_document.md` | Draft model document |
