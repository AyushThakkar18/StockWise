from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FundamentalFeatures:
    revenue_growth: float | None
    net_margin: float | None
    operating_margin: float | None
    return_on_assets: float | None
    liabilities_to_assets: float | None
    cash_to_assets: float | None
    completeness: float


FLOW_CONCEPTS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
}
INSTANT_CONCEPTS = {
    "assets": ("Assets",), "liabilities": ("Liabilities",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
}


def build_fundamental_features(payload: dict, decision_on: date) -> FundamentalFeatures:
    flows = {name: _annual_values(payload, concepts, decision_on) for name, concepts in FLOW_CONCEPTS.items()}
    instant = {name: _latest_value(payload, concepts, decision_on) for name, concepts in INSTANT_CONCEPTS.items()}
    revenue = flows["revenue"][0] if flows["revenue"] else None
    previous_revenue = flows["revenue"][1] if len(flows["revenue"]) > 1 else None
    net_income = flows["net_income"][0] if flows["net_income"] else None
    operating_income = flows["operating_income"][0] if flows["operating_income"] else None
    assets, liabilities, cash = instant["assets"], instant["liabilities"], instant["cash"]
    values = {
        "revenue_growth": _ratio(revenue, previous_revenue, subtract_one=True),
        "net_margin": _ratio(net_income, revenue),
        "operating_margin": _ratio(operating_income, revenue),
        "return_on_assets": _ratio(net_income, assets),
        "liabilities_to_assets": _ratio(liabilities, assets),
        "cash_to_assets": _ratio(cash, assets),
    }
    return FundamentalFeatures(**values, completeness=sum(v is not None for v in values.values()) / len(values))


def _annual_values(payload: dict, concepts: tuple[str, ...], decision_on: date) -> list[float]:
    candidates = []
    for concept in concepts:
        fact = payload.get("facts", {}).get("us-gaap", {}).get(concept, {})
        for items in fact.get("units", {}).values():
            for item in items:
                if item.get("form") != "10-K" or not item.get("filed") or not item.get("end"):
                    continue
                if date.fromisoformat(item["filed"]) > decision_on or "val" not in item:
                    continue
                start = item.get("start")
                if start and not 250 <= (date.fromisoformat(item["end"]) - date.fromisoformat(start)).days <= 380:
                    continue
                candidates.append((item["end"], item["filed"], float(item["val"])))
        if candidates:
            break
    by_period = {}
    for end, filed, value in candidates:
        if end not in by_period or filed > by_period[end][0]:
            by_period[end] = (filed, value)
    latest_periods = sorted(by_period, reverse=True)[:2]
    return [by_period[end][1] for end in latest_periods]


def _latest_value(payload: dict, concepts: tuple[str, ...], decision_on: date) -> float | None:
    candidates = []
    for concept in concepts:
        fact = payload.get("facts", {}).get("us-gaap", {}).get(concept, {})
        for items in fact.get("units", {}).values():
            candidates.extend(
                (item["filed"], item.get("end", ""), float(item["val"]))
                for item in items if item.get("form") in {"10-Q", "10-K"}
                and item.get("filed") and "val" in item
                and date.fromisoformat(item["filed"]) <= decision_on
            )
        if candidates:
            break
    return max(candidates)[2] if candidates else None


def _ratio(left: float | None, right: float | None, subtract_one: bool = False) -> float | None:
    if left is None or right in {None, 0}:
        return None
    value = left / right
    return value - 1 if subtract_one else value
