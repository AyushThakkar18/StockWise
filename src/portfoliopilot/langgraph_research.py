from __future__ import annotations

import operator
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .contracts import Evidence, PolicyCheck
from .evidence import curate_evidence
from .research import RESEARCH_ROLES, ReportRunner, SynthesisRunner, audit_reports
from .research_contracts import (
    CouncilAudit,
    CouncilResult,
    CouncilSynthesis,
    OrchestratedCouncilResult,
    ResearchReport,
    ResearchRole,
)


class CouncilState(TypedDict, total=False):
    symbol: str
    decision_at: datetime
    evidence: tuple[Evidence, ...]
    curated: tuple[Evidence, ...]
    reports: Annotated[list[ResearchReport], operator.add]
    failures: Annotated[list[str], operator.add]
    council: CouncilResult
    synthesis: CouncilSynthesis | None
    route: str


class LangGraphResearchCouncil:
    """Checkpointed five-agent graph with parallel specialists and guarded synthesis."""

    def __init__(
        self, runner: ReportRunner, synthesizer: SynthesisRunner,
        maximum_evidence_age: timedelta = timedelta(days=120),
        minimum_evidence_records: int = 2, checkpointer=None,
    ):
        self.runner = runner
        self.synthesizer = synthesizer
        self.maximum_evidence_age = maximum_evidence_age
        self.minimum_evidence_records = minimum_evidence_records
        self._connection = None
        self.graph = self._build().compile(checkpointer=checkpointer or InMemorySaver())

    @classmethod
    def with_sqlite(
        cls, runner: ReportRunner, synthesizer: SynthesisRunner, path: Path, **kwargs,
    ) -> LangGraphResearchCouncil:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        instance = cls(
            runner, synthesizer, checkpointer=SqliteSaver(connection), **kwargs,
        )
        instance._connection = connection
        return instance

    def evaluate(
        self, symbol: str, decision_at: datetime, evidence: tuple[Evidence, ...], thread_id: str,
    ) -> OrchestratedCouncilResult:
        output = self.graph.invoke(
            {"symbol": symbol, "decision_at": decision_at, "evidence": evidence,
             "reports": [], "failures": []},
            config={"configurable": {"thread_id": thread_id}, "max_concurrency": 4},
        )
        return OrchestratedCouncilResult(
            council=output["council"], synthesis=output.get("synthesis"), route=output["route"],
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _build(self):
        builder = StateGraph(CouncilState)
        builder.add_node("curate", self._curate)
        for role in RESEARCH_ROLES:
            builder.add_node(role.value, self._specialist(role))
            builder.add_edge(role.value, "audit")
        builder.add_node("audit", self._audit)
        builder.add_node("synthesize", self._synthesize)
        builder.add_node("abstain_insufficient", self._abstain_insufficient)
        builder.add_edge(START, "curate")
        builder.add_conditional_edges("curate", self._route_after_curation)
        builder.add_conditional_edges(
            "audit", lambda state: "synthesize" if state["council"].audit.approved else END,
        )
        builder.add_edge("synthesize", END)
        builder.add_edge("abstain_insufficient", END)
        return builder

    def _curate(self, state: CouncilState) -> dict:
        return {"curated": curate_evidence(
            state["evidence"], state["symbol"], state["decision_at"], self.maximum_evidence_age,
        )}

    def _route_after_curation(self, state: CouncilState) -> Sequence[str] | str:
        if len(state["curated"]) < self.minimum_evidence_records:
            return "abstain_insufficient"
        return [role.value for role in RESEARCH_ROLES]

    def _specialist(self, role: ResearchRole):
        def run(state: CouncilState) -> dict:
            try:
                report = self.runner(
                    role, state["symbol"], state["decision_at"], state["curated"],
                )
                return {"reports": [report]}
            except Exception as exc:  # noqa: BLE001 - graph records agent failure and abstains
                return {"failures": [f"{role.value}:{type(exc).__name__}"]}

        return run

    def _audit(self, state: CouncilState) -> dict:
        reports = tuple(sorted(
            state.get("reports", []), key=lambda report: RESEARCH_ROLES.index(report.role),
        ))
        audit = audit_reports(
            state["symbol"], state["decision_at"], state["curated"], reports,
            tuple(state.get("failures", [])),
        )
        return {"council": CouncilResult(
            symbol=state["symbol"], as_of=state["decision_at"],
            evidence_ids=tuple(item.id for item in state["curated"]), reports=reports, audit=audit,
        ), "route": "READY_FOR_SYNTHESIS" if audit.approved else "ABSTAIN_AUDIT_BLOCKED"}

    def _synthesize(self, state: CouncilState) -> dict:
        council = state["council"]
        try:
            synthesis = self.synthesizer(
                state["symbol"], state["decision_at"], council.reports, state["curated"],
            )
        except Exception:  # noqa: BLE001 - synthesis failure cannot reach downstream decisions
            return {"synthesis": None, "route": "ABSTAIN_SYNTHESIS_FAILED"}
        valid = (
            synthesis.symbol == state["symbol"] and synthesis.as_of == state["decision_at"]
            and set(synthesis.report_ids) == {report.report_id for report in council.reports}
            and set(synthesis.evidence_ids).issubset(council.evidence_ids)
        )
        return {
            "synthesis": synthesis if valid else None,
            "route": "SYNTHESIS_COMPLETE" if valid else "ABSTAIN_SYNTHESIS_INVALID",
        }

    def _abstain_insufficient(self, state: CouncilState) -> dict:
        audit = CouncilAudit(
            approved=False,
            checks=(PolicyCheck(name="evidence_completeness", passed=False,
                                detail="insufficient curated evidence"),),
            contradictions=(), blocker_codes=("INSUFFICIENT_EVIDENCE",), evidence_coverage=0,
        )
        return {"council": CouncilResult(
            symbol=state["symbol"], as_of=state["decision_at"],
            evidence_ids=tuple(item.id for item in state["curated"]), reports=(), audit=audit,
        ), "synthesis": None, "route": "ABSTAIN_INSUFFICIENT_EVIDENCE"}
