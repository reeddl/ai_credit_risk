"""Company inputs from SEC XBRL company facts, point in time.

Reads the reduced company-facts files that ``tools/fetch_sec.py`` writes (the
SEC's ``companyfacts`` JSON cut down to the concepts listed in ``CONCEPTS``)
and returns the financial fields of ``ai_pd.features.RAW_COLUMNS`` as they
stood on a given date: only facts filed on or before that date count, and a
later filing of the same period (a restatement) replaces an earlier one.

Every value comes with a provenance row in the format of
``benchmark/data/companies_sources.csv``: the concept, the filing's
accession number and form, the filing date and a link to the filing index.

Rules, stated so a reviewer can challenge them:

- Balance sheet items are read at the latest ``Assets`` date on or before the
  as-of date, and only at that date.
- Cash is cash and equivalents plus current marketable securities.
- Debt is total long-term debt (current and non-current) plus short-term
  borrowings and commercial paper. Leases are left out. A company that tags
  no debt concept at the balance sheet date is taken to have no debt, and its
  provenance row says so.
- Debt due within 24 months is next-twelve-months plus year-two maturities
  from the latest 10-K, plus short-term borrowings and commercial paper; with
  no maturity table, the current portion of debt.
- Hard assets are net property, plant and equipment (with finance-lease
  assets where a company tags them together), plus net oil and gas
  properties for producers.
- Revenue for the year is the trailing twelve months; growth is the latest
  quarter against the same quarter a year earlier; gross margin is the latest
  quarter's. Quarterly flows that are only reported year to date are
  differenced (Q4 is the fiscal year less the first nine months).
- Monthly net burn is minus trailing twelve-month operating cash flow less
  capital expenditure, divided by 12.
- Employees are rarely tagged; when missing they stay missing, which scores
  the labor feature as neutral.
- Only US GAAP facts in US dollars are used. Companies filing in IFRS or in
  another currency are reported as not buildable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

GAAP = "us-gaap"
DEI = "dei"

CONCEPTS = {
    "assets": [(GAAP, "Assets")],
    "cash": [(GAAP, "CashAndCashEquivalentsAtCarryingValue"),
             (GAAP, "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
             (GAAP, "CashAndDueFromBanks"), (GAAP, "Cash")],
    "st_investments": [(GAAP, "ShortTermInvestments"), (GAAP, "MarketableSecuritiesCurrent"),
                       (GAAP, "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
                       (GAAP, "AvailableForSaleSecuritiesCurrent")],
    "debt_total": [(GAAP, "LongTermDebt"), (GAAP, "DebtLongtermAndShorttermCombinedAmount")],
    "debt_noncurrent": [(GAAP, "LongTermDebtNoncurrent"),
                        (GAAP, "LongTermDebtAndCapitalLeaseObligations"),
                        (GAAP, "SeniorLongTermNotes"), (GAAP, "ConvertibleNotesPayable"),
                        (GAAP, "LongTermNotesPayable"), (GAAP, "LongTermLineOfCredit")],
    "debt_current": [(GAAP, "DebtCurrent"), (GAAP, "LongTermDebtCurrent"),
                     (GAAP, "LongTermDebtAndCapitalLeaseObligationsCurrent"),
                     (GAAP, "ConvertibleNotesPayableCurrent"), (GAAP, "NotesPayableCurrent")],
    "short_term_borrowings": [(GAAP, "ShortTermBorrowings")],
    "commercial_paper": [(GAAP, "CommercialPaper")],
    "maturity_next12": [(GAAP, "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths")],
    "maturity_year2": [(GAAP, "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo")],
    "ppe": [(GAAP, "PropertyPlantAndEquipmentNet"),
            (GAAP, "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization")],
    "oil_gas": [(GAAP, "OilAndGasPropertyFullCostMethodNet"),
                (GAAP, "OilAndGasPropertySuccessfulEffortMethodNet")],
    "revenue": [(GAAP, "RevenueFromContractWithCustomerExcludingAssessedTax"), (GAAP, "Revenues"),
                (GAAP, "RevenueFromContractWithCustomerIncludingAssessedTax"),
                (GAAP, "SalesRevenueNet"), (GAAP, "SalesRevenueGoodsNet"),
                (GAAP, "RevenuesNetOfInterestExpense")],
    "gross_profit": [(GAAP, "GrossProfit")],
    "cost_of_revenue": [(GAAP, "CostOfRevenue"), (GAAP, "CostOfGoodsAndServicesSold"),
                        (GAAP, "CostOfGoodsSold"), (GAAP, "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization")],
    "ocf": [(GAAP, "NetCashProvidedByUsedInOperatingActivities"),
            (GAAP, "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "capex": [(GAAP, "PaymentsToAcquirePropertyPlantAndEquipment"),
              (GAAP, "PaymentsToAcquireProductiveAssets"),
              (GAAP, "PaymentsToAcquireOilAndGasPropertyAndEquipment"),
              (GAAP, "PaymentsToAcquireOilAndGasProperty")],
    "employees": [(DEI, "EntityNumberOfEmployees")],
    "shares": [(DEI, "EntityCommonStockSharesOutstanding")],
    "shares_diluted": [(GAAP, "WeightedAverageNumberOfDilutedSharesOutstanding")],
}


def concept_whitelist() -> set[tuple[str, str]]:
    return {c for group in CONCEPTS.values() for c in group}


@dataclass(frozen=True)
class Fact:
    taxonomy: str
    concept: str
    unit: str
    start: date | None
    end: date
    val: float
    accn: str
    form: str
    filed: date

    @property
    def months(self) -> float:
        return 0.0 if self.start is None else (self.end - self.start).days / 30.44

    def link(self, cik: int) -> str:
        return (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{self.accn.replace('-', '')}/{self.accn}-index.htm")


def _d(s: str | None) -> date | None:
    return None if not s else date.fromisoformat(s)


class CompanyFacts:
    def __init__(self, raw: dict):
        self.cik = int(raw["cik"])
        self.name = raw.get("entityName", "")
        self._facts: dict[tuple[str, str], list[Fact]] = {}
        for tax, concepts in raw.get("facts", {}).items():
            for concept, body in concepts.items():
                out = []
                for unit, rows in body.get("units", {}).items():
                    for r in rows:
                        out.append(Fact(tax, concept, unit, _d(r.get("start")), _d(r["end"]),
                                        float(r["val"]), r.get("accn", ""), r.get("form", ""),
                                        _d(r["filed"])))
                self._facts[(tax, concept)] = out

    @classmethod
    def load(cls, path: str | Path) -> "CompanyFacts":
        return cls(json.loads(Path(path).read_text()))

    def has_usd_gaap(self) -> bool:
        return any(f.unit == "USD" for (tax, _), fs in self._facts.items() if tax == GAAP for f in fs)

    def facts(self, key: str, as_of: date, unit: str = "USD") -> list[Fact]:
        """Facts for the first concept in ``CONCEPTS[key]`` order that has any, latest filing per period."""
        for tax, concept in CONCEPTS[key]:
            fs = [f for f in self._facts.get((tax, concept), []) if f.unit == unit and f.filed <= as_of]
            if fs:
                return _latest_per_period(fs)
        return []

    def merged(self, key: str, as_of: date, unit: str = "USD") -> list[Fact]:
        """Facts across every concept in ``CONCEPTS[key]``; an earlier concept wins a shared period."""
        seen: dict[tuple, Fact] = {}
        for tax, concept in CONCEPTS[key]:
            fs = [f for f in self._facts.get((tax, concept), []) if f.unit == unit and f.filed <= as_of]
            for f in _latest_per_period(fs):
                seen.setdefault((f.start, f.end), f)
        return sorted(seen.values(), key=lambda f: (f.end, f.start or f.end))


def _latest_per_period(fs: list[Fact]) -> list[Fact]:
    best: dict[tuple, Fact] = {}
    for f in fs:
        k = (f.start, f.end)
        if k not in best or f.filed > best[k].filed:
            best[k] = f
    return sorted(best.values(), key=lambda f: (f.end, f.start or f.end))


def instant_at(facts: list[Fact], when: date) -> Fact | None:
    hits = [f for f in facts if f.start is None and f.end == when]
    return hits[-1] if hits else None


def latest_instant(facts: list[Fact], as_of: date) -> Fact | None:
    hits = [f for f in facts if f.start is None and f.end <= as_of]
    return max(hits, key=lambda f: (f.end, f.filed)) if hits else None


def _near(a: date, b: date, days: int = 20) -> bool:
    return abs((a - b).days) <= days


def quarter(facts: list[Fact], end: date) -> tuple[float, list[Fact]] | None:
    """Three-month flow ending at ``end``, differencing year-to-date facts if needed."""
    ending = [f for f in facts if f.start is not None and _near(f.end, end, 3)]
    for f in ending:
        if 2.5 <= f.months <= 3.5:
            return f.val, [f]
    for f in sorted(ending, key=lambda f: f.months):
        if f.months < 4:
            continue
        prior = [g for g in facts if g.start == f.start and 2.5 <= f.months - g.months <= 3.5
                 and g.end < f.end]
        if prior:
            g = max(prior, key=lambda g: g.end)
            return f.val - g.val, [f, g]
    return None


def ttm(facts: list[Fact], end: date) -> tuple[float, list[Fact]] | None:
    """Twelve-month flow ending at ``end``.

    A twelve-month fact ending there if one exists; else last fiscal year plus
    this year to date less the same stretch a year earlier; else the sum of
    four quarters.
    """
    for f in facts:
        if f.start is not None and _near(f.end, end, 3) and 11.5 <= f.months <= 12.5:
            return f.val, [f]
    ytd = [f for f in facts if f.start is not None and _near(f.end, end, 3) and 2.5 <= f.months < 11.5]
    if ytd:
        y = max(ytd, key=lambda f: f.months)
        annual = [f for f in facts if f.start is not None and 11.5 <= f.months <= 12.5
                  and _near(f.end, y.start - timedelta(days=1), 20)]
        if annual:
            a = max(annual, key=lambda f: f.filed)
            prior = [f for f in facts if f.start is not None and _near(f.start, a.start, 20)
                     and abs(f.months - y.months) < 0.6]
            if prior:
                p = max(prior, key=lambda f: f.filed)
                return a.val + y.val - p.val, [y, a, p]
    total, used, e = 0.0, [], end
    for i in range(4):
        q = quarter(facts, e)
        if q is None:
            return None
        total += q[0]
        used += q[1]
        if i < 3:
            prev = [f.end for f in facts if f.start is not None and 80 <= (e - f.end).days <= 100]
            if not prev:
                return None
            e = max(prev)
    return total, used


def latest_flow_end(facts: list[Fact], as_of: date) -> date | None:
    ends = [f.end for f in facts if f.start is not None and f.end <= as_of]
    return max(ends) if ends else None


@dataclass
class Built:
    values: dict
    sources: list[dict]
    period_end: date | None
    problems: list[str]


def financial_inputs(cf: CompanyFacts, as_of: date, company_id: str) -> Built:
    """Financial fields of RAW_COLUMNS as they stood on ``as_of``."""
    values, sources, problems = {}, [], []

    def put(field, value, kind, facts=(), note=""):
        values[field] = value
        facts = [f for f in facts if f is not None]
        sources.append({
            "company_id": company_id, "field": field, "value": value, "kind": kind,
            "source": " | ".join(f"SEC XBRL {f.taxonomy}:{f.concept} ({f.form})" for f in facts) or None,
            "url": " | ".join(f.link(cf.cik) for f in facts) or None,
            "date": " | ".join(f.filed.isoformat() for f in facts) or None,
            "note": note or None,
        })

    if not cf.has_usd_gaap():
        return Built({}, [], None, ["no US GAAP facts in US dollars"])

    a = latest_instant(cf.facts("assets", as_of), as_of)
    if a is None:
        return Built({}, [], None, ["no Assets fact on or before the as-of date"])
    bs = a.end
    put("total_assets", a.val, "filing", [a], f"balance sheet {bs}")

    cash = instant_at(cf.facts("cash", as_of), bs)
    sti = instant_at(cf.facts("st_investments", as_of), bs)
    if cash is None:
        problems.append("no cash fact at the balance sheet date")
        put("cash", float("nan"), "missing", note=f"balance sheet {bs}")
    else:
        put("cash", cash.val + (sti.val if sti else 0.0), "filing", [cash, sti],
            "cash and equivalents" + (" + current marketable securities" if sti else ""))

    stb = instant_at(cf.facts("short_term_borrowings", as_of), bs)
    cp = instant_at(cf.facts("commercial_paper", as_of), bs)
    short = (stb.val if stb else 0.0) + (cp.val if cp else 0.0)
    total = instant_at(cf.facts("debt_total", as_of), bs)
    nonc = instant_at(cf.facts("debt_noncurrent", as_of), bs)
    cur = instant_at(cf.facts("debt_current", as_of), bs)
    if total is not None:
        put("debt", total.val + short, "filing", [total, stb, cp], "total long-term debt + short-term borrowings + commercial paper")
    elif nonc is not None or cur is not None:
        extra = short if (cur is None or cur.concept != "DebtCurrent") else 0.0
        put("debt", (nonc.val if nonc else 0.0) + (cur.val if cur else 0.0) + extra, "filing",
            [nonc, cur, stb, cp], "non-current + current debt (+ short-term borrowings and commercial paper where not in current debt)")
    elif short:
        put("debt", short, "filing", [stb, cp], "short-term borrowings and commercial paper only")
    else:
        put("debt", 0.0, "assumed", note=f"no debt concept tagged at the balance sheet date {bs}; taken as no debt")

    k = [f for f in cf.facts("maturity_next12", as_of) if f.start is None]
    n12 = max(k, key=lambda f: f.end) if k else None
    y2 = instant_at(cf.facts("maturity_year2", as_of), n12.end) if n12 else None
    if n12 is not None and (bs - n12.end).days <= 400:
        put("debt_due_24m", n12.val + (y2.val if y2 else 0.0) + short, "filing", [n12, y2, stb, cp],
            f"maturities next 12 months + year two from the filing for {n12.end}, + short-term borrowings and commercial paper")
    elif cur is not None or short:
        extra = short if (cur is None or cur.concept != "DebtCurrent") else 0.0
        put("debt_due_24m", (cur.val if cur else 0.0) + extra, "filing", [cur, stb, cp],
            "current portion only; no maturity table")
    else:
        put("debt_due_24m", 0.0 if values["debt"] == 0 else float("nan"),
            "assumed" if values["debt"] == 0 else "missing", note="no current debt or maturity table")

    ppe = instant_at(cf.facts("ppe", as_of), bs)
    og = instant_at(cf.facts("oil_gas", as_of), bs)
    if ppe is None and og is None:
        problems.append("no PP&E fact at the balance sheet date")
        put("hard_assets", float("nan"), "missing")
    else:
        put("hard_assets", (ppe.val if ppe else 0.0) + (og.val if og else 0.0), "filing", [ppe, og],
            "net PP&E" + (" + net oil and gas properties" if og else ""))

    rev = cf.merged("revenue", as_of)
    end = latest_flow_end(rev, as_of)
    if end is None:
        problems.append("no revenue facts")
        put("revenue_annual", 0.0, "assumed", note="no revenue tagged; taken as none")
        put("revenue_growth", float("nan"), "missing")
        put("gross_margin", float("nan"), "missing")
    else:
        t = ttm(rev, end)
        if t is None:
            problems.append("no trailing twelve-month revenue")
            put("revenue_annual", float("nan"), "missing")
        else:
            put("revenue_annual", t[0], "derived", t[1][:1], f"trailing twelve months to {end}")
        q = quarter(rev, end)
        py_end = end - timedelta(days=365)
        py = [f.end for f in rev if _near(f.end, py_end, 20)]
        q0 = quarter(rev, min(py, key=lambda d: abs((d - py_end).days))) if py else None
        if q and q0 and q0[0] > 0:
            put("revenue_growth", q[0] / q0[0] - 1.0, "derived", [q[1][0], q0[1][0]],
                f"quarter to {end} against the same quarter a year earlier")
        else:
            put("revenue_growth", float("nan"), "missing", note="no comparable quarters")
        gp = quarter(cf.merged("gross_profit", as_of), end)
        cost = quarter(cf.merged("cost_of_revenue", as_of), end)
        if q and q[0] > 0 and gp:
            put("gross_margin", gp[0] / q[0], "derived", [gp[1][0], q[1][0]], f"gross profit / revenue, quarter to {end}")
        elif q and q[0] > 0 and cost:
            put("gross_margin", 1 - cost[0] / q[0], "derived", [cost[1][0], q[1][0]], f"1 - cost of revenue / revenue, quarter to {end}")
        else:
            put("gross_margin", float("nan"), "missing", note="no gross profit or cost of revenue tagged")

    ocf = cf.merged("ocf", as_of)
    capex = cf.merged("capex", as_of)
    cend = latest_flow_end(ocf, as_of)
    o = ttm(ocf, cend) if cend else None
    c = ttm(capex, cend) if cend else None
    if o is None:
        problems.append("no trailing operating cash flow")
        put("monthly_net_burn", float("nan"), "missing")
    else:
        cv = abs(c[0]) if c else 0.0
        put("monthly_net_burn", -(o[0] - cv) / 12.0, "derived", [o[1][0], c[1][0] if c else None],
            f"-(operating cash flow - capex) / 12, twelve months to {cend}" + ("" if c else "; no capex tagged"))

    emp = latest_instant(cf.facts("employees", as_of, unit="pure"), as_of)
    if emp is None:
        put("employees", float("nan"), "missing", note="not tagged in XBRL; labor feature scores neutral")
    else:
        put("employees", emp.val, "filing", [emp])

    return Built(values, sources, bs, problems)


def shares_outstanding(cf: CompanyFacts, as_of: date) -> tuple[float, Fact] | None:
    s = latest_instant(cf.facts("shares", as_of, unit="shares"), as_of)
    if s is not None and (as_of - s.end).days <= 200:
        return s.val, s
    d = cf.merged("shares_diluted", as_of, unit="shares")
    if d:
        f = max(d, key=lambda f: f.end)
        if (as_of - f.end).days <= 200:
            return f.val, f
    return None
