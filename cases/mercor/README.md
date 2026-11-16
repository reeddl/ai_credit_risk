# Case: Mercor

Mercor (founded 2023, San Francisco) runs a marketplace of expert
contractors who produce post-training data and evaluations for AI labs. It
is private and equity funded, with no reported debt.

```bash
python cases/mercor/run_mercor.py
```

Edit `inputs.csv` to change an assumption: `base` is the value used, `low`
and `high` feed the one-at-a-time sensitivity, and every row says where the
figure came from or why it was chosen. Results land in `results/`.

## Public facts used (as of November 16, 2026)

| Item | Figure | Source |
| --- | --- | --- |
| Series D | $750M at $20B, Nov 10, 2026, incl. an employee tender of undisclosed size | Mercor blog; Reuters |
| Series C | $350M at ~$10B, Oct 2025 | TechCrunch, via Contrary and Wikipedia |
| Series B | $100M at $2B, Feb 2025, led by Felicis | TechCrunch, via Contrary |
| Gross run rate | ~$3B, Nov 2026 (CEO post on X, unaudited); ~$2B June 2026 | CellCog summary; TNW; Sacra |
| H1 2026 gross revenue | ~$614M | The Information, via AI Weekly |
| Contractor share of billings | 60-70%, so gross margin ~30-40% | The Information, via TNW and Dealroom |
| Customer concentration | ~91% of H1 2026 revenue from OpenAI and Anthropic | The Information, via AI Weekly |
| Profitability | FCF positive early 2026; ~$6M profit H1 2025 | Sacra; Dealroom (anonymous source) |
| Cash | "more than $500M", mid-2026 (unverified) | ARR Club |
| Staff | 400+ full-time (Sacra, undated); 300 in 2025 (Wikipedia) | |
| Events | March 2026 breach via LiteLLM; Meta paused work; breach class actions; Scale AI trade-secret suit (Sept 2025); contractor misclassification class action (Oct 2025) | Wikipedia; TNW; Contrary |

Not public: cash, net revenue, audited margins, headcount today, any debt,
how the Series D splits between new money and the tender.

## What the model can and cannot say here

- The scorecard is driven by what Mercor clearly has: no debt, reported
  positive cash flow, a fresh raise and very fast growth. Those push the PD
  about 20x below the 9% segment prior.
- Its biggest risk, two unrated labs paying ~91% of revenue, enters only
  through customer concentration and counterparty quality, worth +0.5
  log-odds together. The model has no interaction between concentration and
  the customers' own risk.
- None of the five config scenarios is a customer loss. A demand stall of
  -20% cannot move the revenue score, because growth stays far above the
  top bin. `lose_a_lab` in the run script is a case-specific what-if.
- Labor intensity counts employees, not the 30,000+ weekly contractors, so
  it reads Mercor as extremely capital light. Counting contractors would
  score it 5.
- Weights were never calibrated for private data providers. The CDS
  benchmark found weights too strong for chip and fab names (scale 0.45);
  the same scale here lifts the PD from 0.47% to 2.45%.
- Not captured at all: worker reclassification, breach liability, labs
  bringing data work in-house, synthetic data replacing human data, and the
  chance the next round prices far below $20B. A 90% markdown means a mark
  of $2B, roughly 0.7x today's gross run rate.
