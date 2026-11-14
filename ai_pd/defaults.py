"""The default definition and how observed events are labeled.

A company defaults on the first of four events inside the horizon:

1. a missed debt payment
2. a bankruptcy filing
3. a shutdown
4. a distressed markdown: a sale or a down round that values equity at least
   90% below the last price investors paid (last funding round, or last
   offering for a public company). A down round counts even with no sale.

This is wider than classic credit PD, which counts only losses to lenders.
Most AI companies are funded with equity, so events 3 and 4 are how most of
their failures would show up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class DefaultEvent(str, Enum):
    MISSED_DEBT_PAYMENT = "missed_debt_payment"
    BANKRUPTCY_FILING = "bankruptcy_filing"
    SHUTDOWN = "shutdown"
    DISTRESSED_MARKDOWN = "distressed_markdown"


class MarkdownKind(str, Enum):
    SALE = "sale"
    DOWN_ROUND = "down_round"


@dataclass(frozen=True)
class LabeledEvent:
    company: str
    event: DefaultEvent
    event_date: date
    markdown: float | None = None
    kind: MarkdownKind | None = None
    estimated: bool = False
    source: str = ""


def markdown_share(reference_price: float, event_price: float) -> float:
    """Share of value lost against the reference price (0.9 = down 90%)."""
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    if event_price < 0:
        raise ValueError("event_price cannot be negative")
    return 1.0 - event_price / reference_price


def label_markdown(
    company: str,
    event_date: date,
    reference_price: float,
    event_price: float,
    kind: MarkdownKind,
    *,
    threshold: float = 0.90,
    down_rounds_count: bool = True,
    estimated: bool = False,
    source: str = "",
) -> LabeledEvent | None:
    """Return a default event if this sale or round meets the markdown test.

    Prices are per share. When a private sale price is not disclosed, pass
    the reported deal value and the last post-money valuation instead, and
    set estimated=True so the label is flagged.
    """
    if kind is MarkdownKind.DOWN_ROUND and not down_rounds_count:
        return None
    share = markdown_share(reference_price, event_price)
    # Small tolerance so a price exactly 10% of the reference counts.
    if share + 1e-12 < threshold:
        return None
    return LabeledEvent(
        company=company,
        event=DefaultEvent.DISTRESSED_MARKDOWN,
        event_date=event_date,
        markdown=share,
        kind=kind,
        estimated=estimated,
        source=source,
    )


def first_default(
    events: list[LabeledEvent], start: date, horizon_years: int
) -> dict[str, LabeledEvent]:
    """First default event per company inside [start, start + horizon)."""
    try:
        end = start.replace(year=start.year + horizon_years)
    except ValueError:  # Feb 29 start, non-leap end year
        end = start.replace(year=start.year + horizon_years, day=28)
    first: dict[str, LabeledEvent] = {}
    for ev in sorted(events, key=lambda e: e.event_date):
        if start <= ev.event_date < end and ev.company not in first:
            first[ev.company] = ev
    return first
