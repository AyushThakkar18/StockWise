from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .bounded_agent_strategy import ROLE_WEIGHTS, feature_packet
from .kronos_only_backtest import forecast_score


def council_score(features) -> float:
    return sum(
        ROLE_WEIGHTS[role] * item.signal / 2 * item.confidence
        for role, item in features.items()
    )


def council_approval(features, threshold: float = .15) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    if set(features) != set(ROLE_WEIGHTS):
        reasons.append("INCOMPLETE_COUNCIL")
    else:
        if council_score(features) < threshold:
            reasons.append("INSUFFICIENT_SUPPORT")
        if features["RISK"].signal < 0:
            reasons.append("NEGATIVE_RISK_REVIEW")
        if sum(item.signal > 0 for item in features.values()) < 2:
            reasons.append("NO_POSITIVE_MAJORITY")
        if any(item.signal == -2 for item in features.values()):
            reasons.append("STRONG_OBJECTION")
    return not reasons, tuple(reasons)


@dataclass
class KronosCouncilDecisionEngine:
    candidate_strategy: object
    forecaster: object
    agent: object
    selector: object
    benchmark: tuple
    candidate_count: int = 100
    review_count: int = 50
    maximum_approvals: int = 20
    approval_threshold: float = .15
    decisions: dict[date, dict[str, object]] = field(default_factory=dict)

    def decide(self, history) -> dict[str, object]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        if decision_on in self.decisions:
            return self.decisions[decision_on]
        candidates = tuple(self.candidate_strategy.targets(history))[:self.candidate_count]
        if len(candidates) != self.candidate_count:
            raise ValueError(f"expected {self.candidate_count} deterministic candidates")
        factors = {
            item["symbol"]: item for item in self.candidate_strategy.audits[decision_on]["ranked"]
        }
        market = tuple(bar for bar in self.benchmark if bar.session <= decision_on)
        benchmark_forecast = self.forecaster(market)
        kronos_ranked = []
        print(f"[{decision_on}] Kronos screening 0/{len(candidates)}", flush=True)
        for index, symbol in enumerate(candidates, 1):
            forecast = self.forecaster(history[symbol])
            kronos_ranked.append((symbol, forecast_score(forecast, benchmark_forecast), forecast))
            if index % 10 == 0:
                print(f"[{decision_on}] Kronos screening {index}/{len(candidates)}", flush=True)
        kronos_ranked.sort(key=lambda item: (-item[1], int(factors[item[0]]["rank"]), item[0]))
        reviewed = kronos_ranked[:self.review_count]
        packets = {}
        for symbol, score, forecast in reviewed:
            packet = feature_packet(symbol, history[symbol], self.benchmark, factors[symbol])
            packet["kronos_candidate_forecast"] = forecast.model_dump(mode="json")
            packet["kronos_benchmark_forecast"] = benchmark_forecast.model_dump(mode="json")
            packet["kronos_relative_score"] = score
            packets[symbol] = packet
        features = {symbol: {} for symbol in packets}
        print(f"[{decision_on}] council reviewing Kronos Top {len(packets)}", flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self.agent.evaluate_cross_section, role, packets): role
                for role in ROLE_WEIGHTS
            }
            for future in as_completed(futures):
                role = futures[future]
                for symbol, assessment in future.result().items():
                    features[symbol][role] = assessment
                print(f"[{decision_on}] {role} council review complete", flush=True)
        assessments = {}
        for symbol, _, _ in reviewed:
            approved, reasons = council_approval(features[symbol], self.approval_threshold)
            assessments[symbol] = {
                "approved": approved, "rejection_reasons": reasons,
                "council_score": council_score(features[symbol]),
            }
        synthesis_candidates = [{
            "symbol": symbol, "kronos_rank": index, "kronos_score": score,
            "deterministic_rank": factors[symbol]["rank"],
            "forecast": forecast.model_dump(mode="json"),
            "specialists": {
                role: assessment.model_dump(mode="json")
                for role, assessment in features[symbol].items()
            },
        } for index, (symbol, score, forecast) in enumerate(reviewed, 1)]
        synthesis = self.selector.select(synthesis_candidates)
        selected = list(synthesis.selected_ids[:self.maximum_approvals])
        decision = {
            "decision_date": decision_on.isoformat(),
            "deterministic_universe_count": len(history),
            "deterministic_candidate_count": len(candidates),
            "kronos_review_count": len(reviewed),
            "approved_count": len(selected), "selected": selected,
            "synthesis_rationale": synthesis.rationale,
            "synthesis_abstention_reason": synthesis.abstention_reason,
            "approval_threshold": self.approval_threshold,
            "benchmark_forecast": benchmark_forecast.model_dump(mode="json"),
            "kronos_top_50": [{
                "kronos_rank": index, "symbol": symbol, "kronos_score": score,
                "deterministic_rank": factors[symbol]["rank"],
                "deterministic_score": factors[symbol]["score"],
                "selected": symbol in selected, **assessments[symbol],
                "forecast": forecast.model_dump(mode="json"),
                "agents": {
                    role: assessment.model_dump(mode="json")
                    for role, assessment in features[symbol].items()
                },
            } for index, (symbol, score, forecast) in enumerate(reviewed, 1)],
        }
        self.decisions[decision_on] = decision
        print(
            f"[{decision_on}] council approved {len(selected)}/20: "
            f"{', '.join(selected) if selected else 'none'}", flush=True,
        )
        return decision


@dataclass
class CouncilAllocationStrategy:
    engine: KronosCouncilDecisionEngine
    policy: str
    name: str = "kronos_llm_council_allocation"
    target_history: dict[date, dict[str, Decimal]] = field(default_factory=dict)
    audits: dict[date, dict[str, object]] = field(default_factory=dict)

    def targets(self, history) -> dict[str, Decimal]:
        decision_on = max(bars[-1].session for bars in history.values() if bars)
        decision = self.engine.decide(history)
        selected = list(decision["selected"])
        targets = allocate(selected, self.policy)
        self.target_history[decision_on] = targets
        self.audits[decision_on] = {
            "policy": self.policy, "purchased_count": len(selected),
            "purchased": selected,
            "spy_weight": float(targets.get("SPY", Decimal(0))),
            "cash_weight": float(Decimal(1) - sum(targets.values(), Decimal(0))),
            "targets": {symbol: float(weight) for symbol, weight in targets.items()},
        }
        return targets


def allocate(selected: list[str], policy: str) -> dict[str, Decimal]:
    if len(selected) > 20:
        raise ValueError("at most 20 approvals may be allocated")
    if policy == "equal_capped_cash":
        weight = min(Decimal("0.10"), Decimal(1) / Decimal(len(selected))) if selected else Decimal(0)
        return {symbol: weight for symbol in selected}
    if policy == "five_percent_slots_spy":
        slot = Decimal("0.05")
    elif policy == "half_spy_two_point_five_percent_slots":
        slot = Decimal("0.025")
    else:
        raise ValueError(f"unknown allocation policy {policy}")
    active = slot * Decimal(len(selected))
    return {"SPY": Decimal(1) - active, **{symbol: slot for symbol in selected}}
