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
   for segment-wide effects. Compute glut and demand stall are tied to the
   shared compute demand driver: a cut c to next year's big four
   hyperscaler AI capex estimate (consensus $934.5bn for 2027, as of
   2 November 2026). Demand stall follows that leg; compute glut follows the
   credit-funded leg (neoclouds, labs, sovereigns), cut min(1.5c, 1). Shock
   sizes and severity as written apply at the reference cut of 20% and scale
   in proportion to the leg's cut, floored at −95%. Funding freeze and
   rates up keep fixed sizes.
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

**Results (latest quotes, version 1 weights).**

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

### 6.2 Proposed weight scale for chip and fab segments (November 16, 2026)

Section 6.1 found chip and fab PDs below the credit floor. Each segment can
now carry a `weight_scale` that multiplies every feature weight, so
logit(PD) = logit(prior) + lambda_seg x sum_k w_k z_k + gamma x s. It
changes how far scores move a company from its segment prior without
changing the order of companies inside the segment.

`benchmark/propose_weights.py` fits lambda by least squares so that each
traded chip and fab name lands in the middle (in log-odds) of the band from
the rough physical CDS PD (floor) to the risk-neutral CDS PD (ceiling),
with priors held fixed. The fit is 0.45 (0.43 for chip designers and 0.46
for fab and equipment on their own); any value from 0.25 to 0.65 puts all
six names in the band. The proposal sets 0.45 for both segments, which is
the same as cutting every weight to 45% of its value for them (compute
sensitivity 0.40 to 0.18, runway and funding dependence 0.30 to 0.135, and
so on). Priors stay at 3% and 2%.

| Name | Floor | Ceiling | Model now | Model, scale 0.45 |
| --- | --- | --- | --- | --- |
| Nvidia | 0.46% | 2.29% | 0.18% | 0.85% |
| Intel | 0.41% | 2.04% | 0.34% | 0.90% |
| Broadcom | 0.41% | 2.01% | 0.15% | 0.78% |
| AMD | 0.31% | 1.55% | 0.18% | 0.85% |
| SK Hynix | 0.12% | 0.96% | 0.04% | 0.33% |
| TSMC | 0.12% | 0.93% | 0.05% | 0.39% |

Across all 12 names Spearman moves from 0.77 to 0.78 on the latest quotes
(0.91 to 0.88 for the nine with a fitting segment) and from 0.62 to 0.56 on
the summer quotes. The log-odds slope falls from 1.74 to 1.22.

Moving the priors instead would need 13.9% (chip) and 10.1% (fab), which
would make an average chip designer riskier than an average neocloud (8%),
and would keep fab PDs about 9 times apart from Intel to SK Hynix where
CDS put them about 2 times apart.

Caveats: six names, placeholder risk-premium ratios, and two known feature
problems (collateral ignoring cash, compute sensitivity penalising chip
sellers) that push Nvidia and Broadcom up today. Fixing those would lower
their PDs and call for a refit.

### 6.3 Equity universe and credit tail check (November 17, 2026)

Eight listed names run through the equity engine (`ai_eq`) and the credit
model on the same compute demand driver: Nvidia, AMD, Broadcom, TSMC,
SK Hynix, Oracle, CoreWeave and Vistra. Inputs are in `data/universe/`
with a source row per figure (`stock_sources.csv`, `credit_sources.csv`).
Run `python -m ai_eq.universe --out data/universe/results`.

Inputs. Prices are November 13, 2026 closes. Next-year EPS is the consensus
for the first fiscal year ending 9 to 21 months later, read from free
MarketScreener pages (marked `free_snapshot`; the pages give no date for
their estimates). Operating margin is that year's consensus EBIT margin;
gross margin is trailing twelve months. SK Hynix's consensus EBIT margin is
above its trailing gross margin, so its trailing operating margin is used.
Broadcom's revenue elasticity is the chip-designer band scaled by the AI
share of fiscal 2027 revenue (65.7%); Oracle's is half the neocloud band
(judgment). CoreWeave's consensus EPS is negative through 2028, so it runs
in the credit model only. Vistra's credit inputs were added from its third
quarter 2026 results; the other names reuse the CDS benchmark inputs.

Tail check. For each name and named cut, the credit model's stressed PD (the
worse of demand stall and compute glut at that cut) is set against the
Merton PD with equity moved by the engine's central and high drawdown. A gap
above 1.5 in log-odds (PDs floored at 3 basis points) is flagged, as is a
high drawdown that alone takes equity below 10% of its 52-week high.

Findings on this run (`data/universe/results/headline.md`):

- Central move per 10% cut ranges from -15% (SK Hynix) to -68% (AMD). Most
  of the move comes from the growth revision, not the earnings cut: kappa
  applies to every name at the size fitted on Nvidia, so names priced for
  high growth (AMD at 34x) fall hardest, and SK Hynix (2.3x, below the 5.6x
  floor) moves on earnings alone. Mild-cut moves are likely overstated, as
  noted when kappa was fitted.
- The two models disagree in the tail. At the bust cut the credit model's
  stressed PDs are 0.4% to 11% for the seven profitable names, while the
  equity engine's high case takes AMD, Oracle and TSMC through the 90%
  markdown barrier. Either the credit scenarios are too mild for these
  names or the equity engine is too harsh; the analog fit is too thin to
  say which.
- Merton inputs (market cap, debt, volatility) come from the credit inputs,
  dated November 10 to 13, so the equity move is applied to a market cap a
  few days older than the engine's price for some names.

### 6.4 Real companies and the boom panel (November 17, 2026, in progress)

Version 1 ran on synthetic companies. Step 5 moves it onto listed companies
and fits the financial feature weights on past booms.

- **Real companies.** `data/real/universe.csv` lists 69 names across the 13
  segments. `ai_pd/real.py` builds their inputs from SEC XBRL filings as they
  stood on an as-of date (`ai_pd/xbrl.py`), with one provenance row per input.
  TSMC, SK Hynix and SoftBank file in IFRS or outside the SEC and keep their
  hand-built benchmark rows. Judgment inputs come from the benchmark's hand
  judgments where they exist, otherwise from a segment default marked
  `segment_default`.
- **Boom panel.** `data/analogs/panel_firms.csv` lists 81 firms from four
  post-2009 booms (crypto miners, solar, shale, SPAC-funded EVs). Each 10-K a
  firm filed inside its boom is one firm-year. The label is the model's
  default definition within 12 months: an 8-K reporting bankruptcy (item
  1.03) or a close at or below 10% of the observation-date close. Firm-years
  with no sign the firm was still followed a year later are left out of the
  fit. The `expected` column (failed or survived) was written from memory
  and is only a cross-check, never a label.
- **Fit.** Penalised logistic regression on the six financial features
  (runway, revenue, gross margin, rate sensitivity, leverage, collateral),
  with a separate intercept per boom and a normal prior of sd 0.15 per weight
  centred on today's hand weights. Judgment features have no history and
  keep their hand weights.
- **Hold-out.** Leave one boom out; the crypto miners are the headline. The
  fitted weights are compared with the hand weights on the same held-out
  data. The result is reported as it comes out.
- **Missing inputs.** A ratio with a missing input now scores neutral (3)
  instead of falling back to its zero-denominator value; missing revenue
  growth on recurring revenue scores 3 instead of 4. Results for complete
  inputs are unchanged.

Status: the code, lists and tests are in place. The SEC refuses downloads
from the cloud environment this was built in, so the filings are pulled
locally with `tools/fetch_sec.py` (README, "Running the SEC pull"). Results
go here once that pull is in.

## 7. Parameter log

All parameters live in `config/model.toml`. Each should gain a source, date
and owner before results are used.

| Parameter | Value | Source | Date | Status |
| --- | --- | --- | --- | --- |
| `segments.chip_designer.weight_scale` | 0.45 | CDS benchmark fit, section 6.2 | 2026-11-16 | Proposed |
| `segments.fab_equipment.weight_scale` | 0.45 | CDS benchmark fit, section 6.2 | 2026-11-16 | Proposed |
| `driver.consensus` (driver.toml) | 934.5 ($bn, 2027) | TradingNews summary of company consensus | 2026-11-02 | Sourced, secondary |
| `driver.leg2.beta` (driver.toml) | 1.5 | Judgment | 2026-11-17 | Placeholder |
| `driver.cuts` (driver.toml) | mild 10%, stall 20%, bust 35% | Judgment | 2026-11-17 | Placeholder |
| `driver.credit_reference.cut` (driver.toml) | 20% | Set so version 1 shock sizes are reproduced | 2026-11-17 | Convention |
| `revision.default` (equity.toml) | kappa low 0.64, central 0.93, high 1.22 (growth cut per unit EPS cut) | Fit on Nvidia 2018 and 2022, `data/analogs/fit.md`; Nvidia 2022 fully sourced, Nvidia 2018 peak EPS approximate | 2026-11-17 | Fitted, thin |
| `revision.floor_pe` (equity.toml) | 5.6x forward | Micron's forward P/E at its 2018 peak; peak EPS approximate | 2026-11-17 | Fitted, thin |
| `margin_hit.default` (equity.toml) | low 0.02, central 0.08, high 0.15 | Judgment | 2026-11-17 | Placeholder |
| `segments.*.revenue_elasticity` (equity.toml) | per segment, e.g. chip designers 1.0 / 1.2 / 1.4 | Judgment | 2026-11-17 | Placeholder |
| `valuation` (equity.toml) | r 10%, 4 explicit years, 5-year fade, 3% terminal, 5% earnings floor | Judgment | 2026-11-17 | Placeholder |
| Broadcom `revenue_elasticity` (data/universe/stocks.csv) | 0.66 / 0.79 / 0.92 | Chip-designer band x AI share of FY2027 revenue (65.7%) | 2026-11-17 | Derived |
| Oracle `revenue_elasticity` (data/universe/stocks.csv) | 0.50 / 0.65 / 0.80 | Half the neocloud band | 2026-11-17 | Judgment |
| `PD_FLOOR` (ai_eq/universe.py) | 3 basis points | Basel minimum PD; floor for the tail-check log-odds gap only | 2026-11-17 | Convention |
| `TAU` (ai_pd/fit.py) | 0.15 per weight | Judgment: hand weights move only with clear evidence | 2026-11-17 | Placeholder |
| `ALPHA_SD` (ai_pd/fit.py) | 2.0 log-odds | Weak prior on boom intercepts | 2026-11-17 | Convention |
| `PANEL_JUDGMENTS` (ai_pd/panel.py) | floating debt 25%, liquidation haircut 50% | Judgment; enter rate sensitivity and collateral for panel firms | 2026-11-17 | Placeholder |
| `FOLLOW_DAYS` (ai_pd/panel.py) | 300 days | A filing this long after the observation counts as the firm still followed | 2026-11-17 | Convention |
