# Headline: price move per 10% cut to the compute demand estimate

Inputs as of 2026-11-17 (prices are 2026-11-13 closes). Central case, ranked most exposed first.
Low and high are the engine's band. Credit PDs are 1-year, with default including a 90% markdown.

| # | Name | Leg | Fwd P/E | Per 10% cut (low / central / high) | Stall | Bust | Bust, high case | Credit PD base | Credit PD bust | Merton PD after bust high | Flags at bust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AMD | hyperscaler | 34.4x | -48% / **-68%** / -85% | -91% | -96% | -99% | 0.9% | 2.8% | 100.0% | tails_disagree crosses_90 |
| 2 | Oracle | credit_funded | 13.7x | -41% / **-63%** / -70% | -74% | -86% | -95% | 2.6% | 10.7% | 100.0% | tails_disagree crosses_90 |
| 3 | TSMC | hyperscaler | 15.8x | -37% / **-58%** / -72% | -76% | -85% | -92% | 0.4% | 0.9% | 100.0% | tails_disagree crosses_90 |
| 4 | Nvidia | hyperscaler | 10.4x | -37% / **-54%** / -56% | -62% | -74% | -82% | 0.9% | 2.8% | 39.4% | tails_disagree |
| 5 | Broadcom | hyperscaler | 21.2x | -26% / **-41%** / -57% | -66% | -83% | -85% | 0.8% | 1.7% | 53.1% | tails_disagree |
| 6 | Vistra | hyperscaler | 12.2x | -14% / **-37%** / -61% | -60% | -68% | -78% | 0.1% | 0.4% | 26.5% | tails_disagree |
| 7 | SK Hynix | hyperscaler | 2.3x | -11% / **-15%** / -20% | -30% | -52% | -69% | 0.3% | 0.7% | 13.5% | tails_disagree |
| 8 | CoreWeave | credit_funded | n/a | n/a (loss-making) | n/a | n/a | n/a | 15.5% | 52.7% | n/a | credit_only |

Named cuts: mild 10%, stall 20%, bust 35% to the big four's 2027 AI capex estimate; names on the
credit-funded leg (neocloud segment) take 1.5x the cut. Flags: `tails_disagree` means the credit
model's stressed PD and the Merton PD after the high drawdown differ by more than 1.5 in log-odds;
`crosses_90` means the high drawdown alone takes equity below 10% of its 52-week high;
`credit_only` means consensus EPS is negative, so the equity engine does not run.
