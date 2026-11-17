"""Fit the financial feature weights on the boom panel, starting from the hand weights.

    python -m ai_pd.fit --panel data/analogs/credit_panel/firm_years.csv

Model, for firm-year i in boom b:

    logit P(default within 12 months) = alpha_b + sum_k w_k (score_ik - 3)

over the financial features only (``FITTED``). Each boom gets its own
intercept, so a boom's overall failure rate can't leak into the weights.
The weights are penalised towards today's hand weights with a normal prior
of standard deviation ``TAU`` per weight: with few events they barely move,
with many they follow the data. Intercepts get a weak prior around the
panel's overall default rate.

Hold-out: leave one boom out, fit on the other three, and score the held-out
boom with the fitted weights and the average of the training intercepts. The
same is done with the hand weights held fixed (only intercepts fitted), so
the comparison isolates what the fitted weights add. The crypto miners are
the headline hold-out, as the boom closest to GPU clouds. The result is
reported as it comes out; nothing is tuned to improve it.

Writes a proposal (fit.md, weights.csv, holdout.csv). It does not change
config/model.toml.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig, load_config
from .features import score_features
from .panel import OUT as PANEL_DIR
from .panel import to_raw

FITTED = ["runway", "revenue", "gross_margin", "rate_sensitivity", "leverage", "collateral"]
TAU = 0.15
ALPHA_SD = 2.0
HEADLINE = "crypto_miners"


@dataclass
class Fit:
    weights: pd.Series
    se: pd.Series
    alphas: pd.Series
    n: int
    events: int
    converged: bool


def design(panel: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    scores = score_features(to_raw(panel).reset_index(drop=True), cfg)
    return scores[FITTED] - 3.0


def hand_weights(cfg: ModelConfig) -> pd.Series:
    return pd.Series({k: float(cfg.features[k]["weight"]) for k in FITTED})


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_logit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, w0: pd.Series,
              tau: float = TAU, fix_weights: bool = False, max_iter: int = 100) -> Fit:
    """Penalised logistic regression by Newton's method (MAP estimate)."""
    booms = sorted(set(groups))
    G = np.array([[g == b for b in booms] for g in groups], dtype=float)
    Z = G if fix_weights else np.hstack([G, X.values])
    k = Z.shape[1]
    rate = min(max(y.mean(), 1e-3), 0.5)
    mu = np.r_[np.full(len(booms), np.log(rate / (1 - rate))), [] if fix_weights else w0.values]
    prec = np.r_[np.full(len(booms), 1 / ALPHA_SD**2), [] if fix_weights else np.full(len(w0), 1 / tau**2)]
    offset = X.values @ w0.values if fix_weights else np.zeros(len(y))
    beta = mu.copy()
    converged = False
    for _ in range(max_iter):
        p = _sigmoid(Z @ beta + offset)
        grad = Z.T @ (y - p) - prec * (beta - mu)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(prec)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break
    p = _sigmoid(Z @ beta + offset)
    H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(prec)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    alphas = pd.Series(beta[: len(booms)], index=booms)
    if fix_weights:
        w, wse = w0.copy(), pd.Series(0.0, index=w0.index)
    else:
        w = pd.Series(beta[len(booms):], index=w0.index)
        wse = pd.Series(se[len(booms):], index=w0.index)
    return Fit(w, wse, alphas, len(y), int(y.sum()), converged)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def predict(fit: Fit, X: pd.DataFrame) -> np.ndarray:
    return _sigmoid(fit.alphas.mean() + X.values @ fit.weights.values)


def holdout(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, w0: pd.Series, tau: float = TAU) -> pd.DataFrame:
    rows = []
    for b in sorted(set(groups)):
        tr, te = groups != b, groups == b
        for name, fixed in [("fitted", False), ("hand", True)]:
            f = fit_logit(X[tr], y[tr], groups[tr], w0, tau, fix_weights=fixed)
            p = predict(f, X[te])
            rows.append({"held_out": b, "weights": name, "n": int(te.sum()), "events": int(y[te].sum()),
                         "observed_rate": float(y[te].mean()), "predicted_rate": float(p.mean()),
                         "auc": auc(y[te], p), "brier": float(np.mean((p - y[te]) ** 2))})
    return pd.DataFrame(rows)


def usable(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["followed"].astype(bool)].reset_index(drop=True)


def run(panel: pd.DataFrame, cfg: ModelConfig | None = None, tau: float = TAU):
    cfg = cfg or load_config()
    data = usable(panel)
    X = design(data, cfg)
    y = data["default"].astype(float).values
    g = data["boom"].values
    w0 = hand_weights(cfg)
    full = fit_logit(X, y, g, w0, tau)
    hold = holdout(X, y, g, w0, tau)
    return data, full, hold, w0


def report(panel: pd.DataFrame, data: pd.DataFrame, full: Fit, hold: pd.DataFrame, w0: pd.Series,
           tau: float = TAU) -> str:
    by_boom = data.groupby("boom").agg(firms=("firm_id", "nunique"), firm_years=("firm_id", "size"),
                                       defaults=("default", "sum"), bankruptcies=("bankruptcy", "sum"),
                                       deep_losses=("deep_loss", "sum"))
    lines = ["# Credit weights fitted on the post-2009 booms", "",
             f"{len(data)} followed firm-years ({len(panel) - len(data)} more left out as not followed), "
             f"{full.events} defaults. Weights are penalised towards the hand weights (sd {tau} per weight).",
             "This is a proposal: config/model.toml is unchanged.", "",
             "## Panel", "", "| Boom | Firms | Firm-years | Defaults | Bankruptcy 8-Ks | 90% price falls |",
             "|---|---|---|---|---|---|"]
    for b, r in by_boom.iterrows():
        lines.append(f"| {b} | {r.firms} | {r.firm_years} | {int(r.defaults)} | {int(r.bankruptcies)} | {int(r.deep_losses)} |")
    lines += ["", "## Weights", "", "| Feature | Hand | Fitted | Std error | Moved by |", "|---|---|---|---|---|"]
    for k in w0.index:
        lines.append(f"| {k} | {w0[k]:.2f} | {full.weights[k]:.2f} | {full.se[k]:.2f} | {full.weights[k] - w0[k]:+.2f} |")
    lines += ["", "Judgment features (compute sensitivity, funding dependence, moat, margin path, customers, "
              "counterparty quality, circular financing, labor) have no history and keep their hand weights.", "",
              "## Hold-out (leave one boom out)", "",
              "| Held out | Weights | Firm-years | Defaults | Observed | Predicted | AUC | Brier |",
              "|---|---|---|---|---|---|---|---|"]
    for r in hold.itertuples():
        lines.append(f"| {r.held_out} | {r.weights} | {r.n} | {r.events} | {r.observed_rate:.1%} | "
                     f"{r.predicted_rate:.1%} | {r.auc:.2f} | {r.brier:.3f} |")
    head = hold[hold.held_out == HEADLINE].set_index("weights")
    if len(head) == 2:
        f, h = head.loc["fitted"], head.loc["hand"]
        lines += ["", f"Headline, {HEADLINE} held out: AUC {f.auc:.2f} fitted vs {h.auc:.2f} hand; "
                  f"predicted default rate {f.predicted_rate:.1%} fitted vs {h.predicted_rate:.1%} hand, "
                  f"observed {f.observed_rate:.1%}."]
    if not full.converged:
        lines += ["", "Warning: the full fit did not converge."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fit financial feature weights on the boom panel.")
    ap.add_argument("--panel", type=Path, default=PANEL_DIR / "firm_years.csv")
    ap.add_argument("--out", type=Path, default=PANEL_DIR)
    ap.add_argument("--tau", type=float, default=TAU)
    a = ap.parse_args(argv)
    panel = pd.read_csv(a.panel)
    data, full, hold, w0 = run(panel, tau=a.tau)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"hand": w0, "fitted": full.weights, "se": full.se}).to_csv(a.out / "weights.csv", index_label="feature")
    hold.to_csv(a.out / "holdout.csv", index=False)
    (a.out / "fit.md").write_text(report(panel, data, full, hold, w0, a.tau))
    print((a.out / "fit.md").read_text())


if __name__ == "__main__":
    main()
