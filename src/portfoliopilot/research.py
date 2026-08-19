from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from .contracts import Evidence, PolicyCheck
from .evidence import curate_evidence
from .research_contracts import (
    Contradiction,
    CouncilAudit,
    CouncilResult,
    CouncilSynthesis,
    FindingStance,
    OrchestratedCouncilResult,
    ResearchReport,
    ResearchRole,
    Severity,
)

ReportRunner = Callable[[ResearchRole, str, datetime, tuple[Evidence, ...]], ResearchReport]
SynthesisRunner = Callable[[str, datetime, tuple[ResearchReport, ...], tuple[Evidence, ...]], CouncilSynthesis]

RESEARCH_ROLES = (
    ResearchRole.BUSINESS_CHANGE,
    ResearchRole.CATALYST_EVENT,
    ResearchRole.FAILURE_MODE,
    ResearchRole.PORTFOLIO_CONTEXT,
)


class ResearchCouncil:
    def __init__(
        self, runner: ReportRunner, maximum_evidence_age: timedelta = timedelta(days=120),
        minimum_evidence_records: int = 2,
    ):
        self.runner = runner
        self.maximum_evidence_age = maximum_evidence_age
        self.minimum_evidence_records = minimum_evidence_records

    def evaluate(
        self, symbol: str, decision_at: datetime, evidence: tuple[Evidence, ...]
    ) -> CouncilResult:
        curated = curate_evidence(evidence, symbol, decision_at, self.maximum_evidence_age)
        if len(curated) < self.minimum_evidence_records:
            audit = CouncilAudit(
                approved=False,
                checks=(PolicyCheck(
                    name="evidence_completeness", passed=False,
                    detail=f"requires {self.minimum_evidence_records} usable records",
                ),),
                contradictions=(), blocker_codes=("INSUFFICIENT_EVIDENCE",),
                evidence_coverage=0.0,
            )
            return CouncilResult(symbol=symbol, as_of=decision_at, evidence_ids=tuple(item.id for item in curated), reports=(), audit=audit)
        reports = []
        failures = []
        for role in RESEARCH_ROLES:
            try:
                # Each role receives only curated source evidence, never another role's conclusions.
                reports.append(self.runner(role, symbol, decision_at, curated))
            except Exception as exc:  # noqa: BLE001 - external model boundary must fail closed
                failures.append(f"{role.value}:{type(exc).__name__}")
        audit = audit_reports(symbol, decision_at, curated, tuple(reports), tuple(failures))
        return CouncilResult(
            symbol=symbol, as_of=decision_at, evidence_ids=tuple(item.id for item in curated),
            reports=tuple(reports), audit=audit,
        )


class OrchestratedResearchCouncil(ResearchCouncil):
    """Parallel specialist fan-out, deterministic audit gate, then synthesis-agent fan-in."""

    def __init__(self, runner: ReportRunner, synthesizer: SynthesisRunner, **kwargs):
        super().__init__(runner, **kwargs)
        self.synthesizer = synthesizer

    def evaluate_orchestrated(
        self, symbol: str, decision_at: datetime, evidence: tuple[Evidence, ...]
    ) -> OrchestratedCouncilResult:
        curated = curate_evidence(evidence, symbol, decision_at, self.maximum_evidence_age)
        if len(curated) < self.minimum_evidence_records:
            council = super().evaluate(symbol, decision_at, evidence)
            return OrchestratedCouncilResult(
                council=council, synthesis=None, route="ABSTAIN_INSUFFICIENT_EVIDENCE",
            )
        reports, failures = [], []
        with ThreadPoolExecutor(max_workers=len(RESEARCH_ROLES)) as executor:
            futures = {
                executor.submit(self.runner, role, symbol, decision_at, curated): role
                for role in RESEARCH_ROLES
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    reports.append(future.result())
                except Exception as exc:  # noqa: BLE001 - external agent boundary
                    failures.append(f"{role.value}:{type(exc).__name__}")
        ordered = tuple(sorted(reports, key=lambda report: RESEARCH_ROLES.index(report.role)))
        audit = audit_reports(symbol, decision_at, curated, ordered, tuple(failures))
        council = CouncilResult(
            symbol=symbol, as_of=decision_at, evidence_ids=tuple(item.id for item in curated),
            reports=ordered, audit=audit,
        )
        if not audit.approved:
            return OrchestratedCouncilResult(
                council=council, synthesis=None, route="ABSTAIN_AUDIT_BLOCKED",
            )
        try:
            synthesis = self.synthesizer(symbol, decision_at, ordered, curated)
        except Exception:  # noqa: BLE001 - synthesis agent must fail closed
            return OrchestratedCouncilResult(
                council=council, synthesis=None, route="ABSTAIN_SYNTHESIS_FAILED",
            )
        expected_reports = {report.report_id for report in ordered}
        expected_evidence = {item.id for item in curated}
        if (
            synthesis.symbol != symbol or synthesis.as_of != decision_at
            or set(synthesis.report_ids) != expected_reports
            or not set(synthesis.evidence_ids).issubset(expected_evidence)
        ):
            return OrchestratedCouncilResult(
                council=council, synthesis=None, route="ABSTAIN_SYNTHESIS_INVALID",
            )
        return OrchestratedCouncilResult(
            council=council, synthesis=synthesis, route="SYNTHESIS_COMPLETE",
        )


def audit_reports(
    symbol: str, decision_at: datetime, evidence: tuple[Evidence, ...],
    reports: tuple[ResearchReport, ...], runner_failures: tuple[str, ...] = (),
    minimum_sources: int = 2, minimum_coverage: float = 0.75,
) -> CouncilAudit:
    evidence_ids = {item.id for item in evidence}
    required_roles = set(RESEARCH_ROLES)
    received_roles = {report.role for report in reports}
    checks = [
        PolicyCheck(
            name="all_roles_present", passed=received_roles == required_roles,
            detail="all independent roles required",
        ),
        PolicyCheck(
            name="report_identity",
            passed=all(report.symbol == symbol and report.as_of == decision_at for report in reports),
            detail="symbol and decision timestamp must match",
        ),
        PolicyCheck(
            name="evidence_references",
            passed=all(
                set(finding.evidence_ids).issubset(evidence_ids)
                for report in reports for finding in report.findings
            ),
            detail="every factual finding must reference curated evidence",
        ),
    ]
    referenced = {
        evidence_id for report in reports for finding in report.findings
        for evidence_id in finding.evidence_ids if evidence_id in evidence_ids
    }
    coverage = len(referenced) / len(evidence_ids) if evidence_ids else 0.0
    checks.extend((
        PolicyCheck(
            name="source_diversity", passed=len({item.source for item in evidence}) >= minimum_sources,
            detail=f"requires evidence from at least {minimum_sources} distinct sources",
        ),
        PolicyCheck(
            name="evidence_coverage", passed=coverage >= minimum_coverage,
            detail=f"requires at least {minimum_coverage:.0%} citation coverage",
        ),
    ))
    contradictions = detect_contradictions(reports)
    high_unresolved = any(
        item.severity == Severity.HIGH and not item.resolved for item in contradictions
    )
    checks.append(PolicyCheck(
        name="high_severity_contradictions", passed=not high_unresolved,
        detail="unresolved high-severity contradictions block approval",
    ))
    blockers = list(runner_failures)
    blockers.extend(blocker for report in reports for blocker in report.blockers)
    if high_unresolved:
        blockers.append("UNRESOLVED_HIGH_CONTRADICTION")
    if not all(check.passed for check in checks):
        blockers.append("AUDIT_CHECK_FAILED")
    return CouncilAudit(
        approved=all(check.passed for check in checks) and not blockers,
        checks=tuple(checks), contradictions=contradictions,
        blocker_codes=tuple(sorted(set(blockers))), evidence_coverage=coverage,
    )


def detect_contradictions(reports: tuple[ResearchReport, ...]) -> tuple[Contradiction, ...]:
    topics = {finding.topic.strip().casefold() for report in reports for finding in report.findings}
    contradictions = []
    for topic in sorted(topics):
        supporting = {
            report.report_id for report in reports for finding in report.findings
            if finding.topic.strip().casefold() == topic and finding.stance == FindingStance.SUPPORT
        }
        concerning = {
            report.report_id for report in reports for finding in report.findings
            if finding.topic.strip().casefold() == topic and finding.stance == FindingStance.CONCERN
        }
        if supporting and concerning:
            severities = [
                finding.severity for report in reports for finding in report.findings
                if finding.topic.strip().casefold() == topic
            ]
            severity = Severity.HIGH if Severity.HIGH in severities else Severity.MEDIUM
            contradictions.append(Contradiction(
                topic=topic, supporting_report_ids=tuple(sorted(supporting)),
                concerning_report_ids=tuple(sorted(concerning)), severity=severity,
            ))
    return tuple(contradictions)
