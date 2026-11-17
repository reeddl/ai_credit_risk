"""Fit the growth revision on analog drawdowns and write data/analogs/fit.md.

python -m ai_eq.calibrate [--out data/analogs/fit.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_eq.config import load_equity_config
from ai_eq.revision import DATA, fit, fit_episode, load_episodes, load_sources


def sensitivity(episodes, sources, cfg) -> list[str]:
    """kappa when an approximate peak EPS is 10% lower or higher."""
    lines = []
    approx = sources[(sources["field"] == "peak_eps") & (sources["status"] == "approximate")]["episode"]
    for _, row in episodes[(episodes["role"] == "kappa") & episodes["episode"].isin(approx)].iterrows():
        cells = []
        for m in (0.9, 1.0, 1.1):
            r = row.copy()
            r["peak_eps"] = float(row["peak_eps"]) * m
            f = fit_episode(r, cfg)
            cells.append(f"{f.kappa:.2f}")
        lines.append(f"| {row['episode']} | " + " | ".join(cells) + " |")
    if not lines:
        return []
    return ["The peak EPS figures below are approximate, and kappa is sensitive to them:", "",
            "| Episode | Peak EPS -10% | As used | Peak EPS +10% |", "| --- | --- | --- | --- |", *lines, ""]


def report(result: dict, episodes, sources, cfg=None) -> str:
    status = sources.pivot_table(index="episode", columns="field", values="status", aggfunc="first")
    lines = ["# Growth revision fit on analog drawdowns", "",
             "Written by `python -m ai_eq.calibrate`. Figures and their sources are in",
             "`episodes.csv` and `episode_sources.csv`; status per figure is sourced, derived",
             "(arithmetic on sourced figures) or approximate (no dated page found).", "",
             "| Episode | Role | Price change | EPS cut | Implied growth, peak | Implied growth, later | kappa | Approximate figures |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in result["episodes"]:
        approx = [c for c in status.columns if status.loc[f.episode, c] == "approximate"] if f.episode in status.index else []
        lines.append(f"| {f.episode} | {f.role} | {f.price_change:+.0%} | {f.eps_cut:.0%} | {f.g_peak:+.1%} | "
                     f"{f.g_later:+.1%} | {f.kappa:.2f} | {', '.join(approx) or 'none'} |")
    for _, row in episodes[episodes["role"] == "excluded"].iterrows():
        lines.append(f"| {row['episode']} | excluded | | | | | | {row['exclude_reason']} |")
    k = result["kappa"]
    lines += ["", f"kappa: low {k['low']:.2f}, central {k['central']:.2f}, high {k['high']:.2f}.",
              f"floor_pe: {result['floor_pe']:.1f}x forward.", ""]
    if cfg is not None:
        lines += sensitivity(episodes, sources, cfg)
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DATA / "fit.md"))
    a = p.parse_args(argv)
    episodes, sources, cfg = load_episodes(), load_sources(), load_equity_config()
    text = report(fit(episodes, cfg), episodes, sources, cfg)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
