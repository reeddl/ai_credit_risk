# Model document (draft): AI company default model, version 1

Status: draft. Inputs are synthetic and every parameter is a placeholder.
This document is written so a model risk reviewer can challenge each part.

## 1. Purpose and use

The model estimates a 1-year (headline) and 3-year probability of default
for companies whose business depends mainly on AI, and how that probability
moves under named stress scenarios. Intended use: a documented, challengeable
view of relative and stressed risk across the AI stack. It is not yet fit
for pricing, limits or capital.

## 2. Target definition

A company defaults on the first of four events within the horizon:

| Event | Evidence |
| --- | --- |
| Missed debt payment | Filings, lender notices, press |
| Bankruptcy filing | Court dockets, filings |
| Shutdown | Company announcement, press |
| Distressed markdown: a sale or a down round at least 90% below the last price investors paid | Sale or round price per share against the last round or offering price |

The target is wider than classic credit PD, which counts only losses to
lenders. It is closer to the chance of a default or a near-total loss to
equity holders. The label PD is kept for familiarity.

Markdown labeling: the event date is the day the sale or round is announced.
Where a private price is not disclosed, the markdown is estimated from the
reported deal value against the last post-money valuation, and the label is
flagged as estimated. Open item: confirm the reference price (last round or
offering price, the current default, versus a peak value).

## 3. Segmentation

Thirteen segments, each with its own prior PD and scenario sensitivities:
closed model providers, open-weight model providers, app makers, data
providers (strong and weak moat), neoclouds and data centers, power
suppliers, chip designers, fabs and equipment, networking and cooling,
infrastructure software, robotics, and GPU lessors. Hyperscalers are out of
scope as borrowers but matter as counterparties.

## 4. Method

1. **Segment prior.** 1-year PD per segment, from analog episodes and expert
   judgment. Placeholder values in v1.
2. **Scorecard.** Fourteen features, each scored 1 to 5 from a raw metric
   with fixed bins, enter as weight × (score − 3) on the log-odds scale. A
   company scoring 3 on every feature keeps its segment prior. Missing
   metrics score 3 (neutral) and should be flagged in production.
3. **3-year PD.** Derived from the 1-year PD with a constant hazard.
4. **Merton cross-check.** Equity is valued as a call on assets. The default
   barrier is the higher of debt and the asset value at which equity falls
   to 10% of the last price paid. Simplifications: V0 ≈ E + D e^(−rT),
   σ_V = σ_E E / V0, zero asset drift. Gaps above 1.5 log-odds are flagged.
5. **Scenarios.** Each shocks raw inputs (GPU prices and utilisation,
   revenue and growth, equity marks, rates) and adds γ_segment × severity
   for segment-wide effects.
6. **Calibration.** PDs map to implied grades using placeholder boundaries
   to be replaced with a published agency default study.

## 5. Known limitations

- No in-sample AI defaults; priors, weights and shocks are judgment.
- The prior is the PD of a company that scores 3 on every feature, not the
  segment's average PD. A segment whose typical company scores well sits
  below its prior.
- Private-company inputs need heavy estimation.
- Analog episodes may not describe an AI downturn.
- The Merton layer uses simplified asset value and volatility.

## 6. Validation plan

Conceptual soundness write-up per layer; benchmarking against agency ratings
and spreads where they exist; an analog back-test (dot-com, telecom, crypto
miners) labeled with the same four events; one-at-a-time sensitivity (in
v1); stability under input perturbation; a log of every judgment input with
source, date and owner; quarterly monitoring.

## 7. Parameter log

All parameters live in `config/model.toml`. Each should gain a source, date
and owner before results are used.
