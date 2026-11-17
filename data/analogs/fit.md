# Growth revision fit on analog drawdowns

Written by `python -m ai_eq.calibrate`. Figures and their sources are in
`episodes.csv` and `episode_sources.csv`; status per figure is sourced, derived
(arithmetic on sourced figures) or approximate (no dated page found).

| Episode | Role | Price change | EPS cut | Implied growth, peak | Implied growth, later | kappa | Approximate figures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nvidia_2018 | kappa | -55% | 12% | +23.8% | +8.9% | 1.22 | peak_eps |
| nvidia_2022 | kappa | -64% | 26% | +37.6% | +20.8% | 0.64 | none |
| micron_2018 | floor | -34% | 33% | -17.6% | -17.8% | 0.01 | peak_eps |
| cisco_2000 | excluded | | | | | | No same-year consensus EPS found for the peak; the 2001 multiple (over 30x) is on a later fiscal year |
| nvidia_2026 | excluded | | | | | | Estimates rose while the stock fell (forward P/E 20.8x to 13.5x); a pure multiple move with no earnings cut cannot pin down a revision per unit cut |

kappa: low 0.64, central 0.93, high 1.22.
floor_pe: 5.6x forward.

The peak EPS figures below are approximate, and kappa is sensitive to them:

| Episode | Peak EPS -10% | As used | Peak EPS +10% |
| --- | --- | --- | --- |
| nvidia_2018 | 6.87 | 1.22 | 0.63 |
