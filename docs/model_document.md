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

### 6.1 CDS benchmark (November 14, 2026)

**What was compared.** Twelve companies with traded single-name CDS:
CoreWeave, Oracle, Amazon (neocloud), Nvidia, AMD, Broadcom (chip
designer), Intel, TSMC, SK Hynix (fab and equipment), Corning
(networking), Meta (closed model) and SoftBank (GPU lessor). Meta, Amazon
and SoftBank are poor segment fits: hyperscalers and holding companies
have no segment of their own. Model inputs come from each company's latest
quarterly report; CDS quotes are 5-year levels from free provider
summaries, wire reports and press (`benchmark/data/cds_quotes.csv`).

**Method.** Credit triangle with 40% recovery: hazard = spread / (1 - R),
PD(t) = 1 - exp(-hazard x t). CDS PDs are risk-neutral; a rough physical
estimate divides the hazard by an illustrative risk-premium ratio (8x for
the tightest spreads down to 1.5x for distressed names, after the pattern
in Hull, Predescu and White, 2005). CDS cover credit events only, while
this model's default event also counts shutdowns, 90% distressed sales and
90% down rounds, so for names with traded credit the physical CDS PD is a
floor for the model PD, not a target. Ranking is measured with Spearman and
Kendall correlations and a permutation p-value; level with the model / CDS
PD ratio and the slope of model log-odds on CDS log-odds.

**Results (latest quotes).**

| Sample | n | Spearman | Permutation p | Median model / RN PD | Log-odds slope |
| --- | --- | --- | --- | --- | --- |
| All names | 12 | 0.77 | 0.004 | 0.12x | 1.74 |
| Good or approximate fit | 9 | 0.91 | 0.002 | 0.11x | 1.97 |

The summer 2026 snapshot (June to August quotes) gives Spearman 0.62 for
all names, with the same level pattern.

**Findings.**

1. The model ranks traded names about as the market does, best among names
   whose segment fits.
2. Its PDs are about twice as dispersed as the market's in log-odds terms.
   CoreWeave (15.6% vs 17.5% risk-neutral) is close; investment-grade chip
   and fab names sit at 0.04% to 0.34%, below even the rough physical
   credit floor (0.12% to 0.46%). Either the feature weights are too large
   overall or the priors for strong segments are too low. The floor implies
   priors of at least about 4% (fab and equipment) and 7% (chip designer),
   against 2% and 3% today, if the weights stay as they are. Shrinking the
   weights is the alternative; a weight-scale parameter is the obvious next
   calibration step.
3. Feature issues that real data exposed:
   - Collateral scores cash-rich, asset-light firms worst (Nvidia, Broadcom
     score 5) because cash is not counted as collateral.
   - Compute sensitivity penalises chip sellers (Nvidia scores 4) as if
     falling compute prices hurt them like they hurt GPU lessors.
   - Runway reads hyperscaler capex as burn (Amazon about $11B a month).
   - Hyperscalers and investment holding companies need their own segment.
   - In the combined scenario the compute glut lowers compute buyers'
     costs, so combined stress can be milder than demand stall alone.
4. The Merton layer is very sensitive to the reference price. This
   benchmark uses the 52-week high, which makes recent sell-offs (Intel,
   Corning, Oracle) look like approaches to the 90% markdown barrier.

**What it cannot test.** Private model labs, app makers, data providers and
most neoclouds have no traded credit, so their PDs, which the model exists
for, are not benchmarked. No free CDS levels were found for Microsoft,
Alphabet, Micron, power names (Vistra, NRG, Constellation, Talen), Equinix
or Digital Realty, and no clean free bond spreads were found to stand in.
Many inputs are secondary-source figures, 24-month maturities are often
only the current portion, and some volatilities are 30-day implied.

**Open choices for the model owner.** Reference price for public companies
(52-week high here), recovery rate (40%), the risk-neutral to physical
ratios, and the segment for hyperscalers and holding companies.

## 7. Parameter log

All parameters live in `config/model.toml`. Each should gain a source, date
and owner before results are used.
