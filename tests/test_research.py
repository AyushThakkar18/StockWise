from datetime import UTC, datetime, timedelta

from portfoliopilot.contracts import Evidence, Quality
from portfoliopilot.evidence import curate_evidence
from portfoliopilot.langgraph_research import LangGraphResearchCouncil
from portfoliopilot.research import RESEARCH_ROLES, OrchestratedResearchCouncil, ResearchCouncil
from portfoliopilot.research_contracts import (
    CouncilSynthesis,
    CouncilVerdict,
    FindingStance,
    ResearchFinding,
    ResearchReport,
    ResearchRole,
    Severity,
)

NOW = datetime(2025, 1, 10, 21, tzinfo=UTC)


def evidence(item_id: str, available: datetime | None = None, claim: str | None = None) -> Evidence:
    timestamp = available or NOW - timedelta(days=1)
    return Evidence(
        id=item_id, symbol="ABC", claim=claim or f"claim {item_id}",
        source="filing" if item_id.endswith("1") else "news",
        observed_at=timestamp, published_at=timestamp, available_to_strategy_at=timestamp,
        retrieved_at=max(timestamp, NOW - timedelta(hours=1)), vintage="v1", quality=Quality.PASS,
    )


def finding(topic: str, stance: FindingStance, evidence_id: str, severity=Severity.LOW):
    return ResearchFinding(
        topic=topic, factual_claim="A source record states a material fact.",
        interpretation="The fact may affect the monitored thesis.", stance=stance,
        severity=severity, evidence_ids=(evidence_id,),
        uncertainty_notes="The duration and magnitude are not established.",
    )


def good_runner(role: ResearchRole, symbol: str, as_of: datetime, records: tuple[Evidence, ...]):
    index = RESEARCH_ROLES.index(role) % len(records)
    return ResearchReport(
        report_id=f"r:{role}", role=role, symbol=symbol, as_of=as_of,
        findings=(finding(role.value, FindingStance.NEUTRAL, records[index].id),),
        model="fixture", prompt_version="v1",
    )


def test_curator_rejects_future_stale_and_duplicate_records() -> None:
    current = evidence("e1", claim="Same Claim")
    duplicate = evidence("e2", claim=" same claim ").model_copy(update={"source": current.source})
    future = evidence("future", NOW + timedelta(minutes=1))
    stale = evidence("stale", NOW - timedelta(days=200))
    curated = curate_evidence((current, duplicate, future, stale), "ABC", NOW, timedelta(days=30))
    assert tuple(item.id for item in curated) == ("e1",)


def test_complete_independent_reports_pass_deterministic_audit() -> None:
    records = (evidence("e1"), evidence("e2"))
    result = ResearchCouncil(good_runner).evaluate("ABC", NOW, records)
    assert result.audit.approved
    assert {report.role for report in result.reports} == set(RESEARCH_ROLES)
    assert result.audit.evidence_coverage == 1


def test_runner_failure_causes_abstention() -> None:
    def runner(role, symbol, as_of, records):
        if role == ResearchRole.FAILURE_MODE:
            raise TimeoutError("provider unavailable")
        return good_runner(role, symbol, as_of, records)

    result = ResearchCouncil(runner).evaluate("ABC", NOW, (evidence("e1"), evidence("e2")))
    assert not result.audit.approved
    assert any("FAILURE_MODE:TimeoutError" == code for code in result.audit.blocker_codes)


def test_unknown_evidence_reference_blocks_report() -> None:
    def runner(role, symbol, as_of, records):
        report = good_runner(role, symbol, as_of, records)
        if role == ResearchRole.BUSINESS_CHANGE:
            return report.model_copy(update={"findings": (
                finding("unknown", FindingStance.NEUTRAL, "invented-source"),
            )})
        return report

    result = ResearchCouncil(runner).evaluate("ABC", NOW, (evidence("e1"), evidence("e2")))
    assert not result.audit.approved
    assert not next(check for check in result.audit.checks if check.name == "evidence_references").passed


def test_single_source_echo_chamber_blocks_council_approval() -> None:
    records = tuple(
        item.model_copy(update={"source": "same-source"})
        for item in (evidence("e1"), evidence("e2"))
    )
    result = ResearchCouncil(good_runner).evaluate("ABC", NOW, records)
    assert not result.audit.approved
    assert not next(check for check in result.audit.checks if check.name == "source_diversity").passed


def test_high_severity_contradiction_blocks_action() -> None:
    def runner(role, symbol, as_of, records):
        stance = FindingStance.SUPPORT if role == ResearchRole.BUSINESS_CHANGE else FindingStance.CONCERN
        return ResearchReport(
            report_id=f"r:{role}", role=role, symbol=symbol, as_of=as_of,
            findings=(finding("leverage", stance, records[0].id, Severity.HIGH),),
            model="fixture", prompt_version="v1",
        )

    result = ResearchCouncil(runner).evaluate("ABC", NOW, (evidence("e1"), evidence("e2")))
    assert not result.audit.approved
    assert result.audit.contradictions[0].severity == Severity.HIGH
    assert "UNRESOLVED_HIGH_CONTRADICTION" in result.audit.blocker_codes


def test_insufficient_evidence_does_not_call_model() -> None:
    called = False

    def runner(*args):
        nonlocal called
        called = True
        return good_runner(*args)

    result = ResearchCouncil(runner).evaluate("ABC", NOW, (evidence("e1"),))
    assert not result.audit.approved
    assert not called


def synthesis(symbol, as_of, reports, records):
    return CouncilSynthesis(
        symbol=symbol, as_of=as_of, verdict=CouncilVerdict.SUPPORT,
        summary="Independent reports support continued research under stated risks.",
        report_ids=tuple(report.report_id for report in reports),
        evidence_ids=tuple(record.id for record in records), risk_flags=(),
        model="fixture", prompt_version="synthesis-v1",
    )


def test_orchestrator_fans_out_specialists_and_routes_to_synthesis() -> None:
    result = OrchestratedResearchCouncil(good_runner, synthesis).evaluate_orchestrated(
        "ABC", NOW, (evidence("e1"), evidence("e2")),
    )
    assert result.route == "SYNTHESIS_COMPLETE"
    assert result.synthesis is not None
    assert len(result.council.reports) == 4


def test_orchestrator_does_not_synthesize_when_specialist_audit_fails() -> None:
    called = False

    def broken_runner(role, symbol, as_of, records):
        if role == ResearchRole.FAILURE_MODE:
            raise TimeoutError
        return good_runner(role, symbol, as_of, records)

    def forbidden_synthesis(*args):
        nonlocal called
        called = True
        return synthesis(*args)

    result = OrchestratedResearchCouncil(
        broken_runner, forbidden_synthesis,
    ).evaluate_orchestrated("ABC", NOW, (evidence("e1"), evidence("e2")))
    assert result.route == "ABSTAIN_AUDIT_BLOCKED"
    assert result.synthesis is None
    assert not called


def test_langgraph_runs_parallel_council_and_synthesis_graph() -> None:
    graph = LangGraphResearchCouncil(good_runner, synthesis)
    result = graph.evaluate(
        "ABC", NOW, (evidence("e1"), evidence("e2")), thread_id="successful-council",
    )
    assert result.route == "SYNTHESIS_COMPLETE"
    assert result.synthesis is not None
    assert len(result.council.reports) == 4


def test_langgraph_sqlite_checkpoints_and_fail_closed_route(tmp_path) -> None:
    graph = LangGraphResearchCouncil.with_sqlite(
        good_runner, synthesis, tmp_path / "council-checkpoints.db",
    )
    try:
        result = graph.evaluate("ABC", NOW, (evidence("e1"),), thread_id="insufficient")
        assert result.route == "ABSTAIN_INSUFFICIENT_EVIDENCE"
        assert result.synthesis is None
        assert (tmp_path / "council-checkpoints.db").exists()
    finally:
        graph.close()
