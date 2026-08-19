from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from .contracts import FrozenModel, PolicyCheck


class ResearchRole(StrEnum):
    BUSINESS_CHANGE = "BUSINESS_CHANGE"
    CATALYST_EVENT = "CATALYST_EVENT"
    FAILURE_MODE = "FAILURE_MODE"
    PORTFOLIO_CONTEXT = "PORTFOLIO_CONTEXT"


class FindingStance(StrEnum):
    SUPPORT = "SUPPORT"
    CONCERN = "CONCERN"
    NEUTRAL = "NEUTRAL"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ResearchFinding(FrozenModel):
    topic: str = Field(min_length=1, max_length=100)
    factual_claim: str = Field(min_length=1, max_length=500)
    interpretation: str = Field(min_length=1, max_length=500)
    stance: FindingStance
    severity: Severity
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    uncertainty_notes: str = Field(min_length=1, max_length=300)


class ResearchReport(FrozenModel):
    report_id: str
    role: ResearchRole
    symbol: str
    as_of: datetime
    findings: tuple[ResearchFinding, ...]
    blockers: tuple[str, ...] = ()
    model: str
    prompt_version: str

    @model_validator(mode="after")
    def failure_role_must_be_independent(self) -> ResearchReport:
        if not self.report_id or not self.prompt_version:
            raise ValueError("report identity and prompt version are required")
        return self


class Contradiction(FrozenModel):
    topic: str
    supporting_report_ids: tuple[str, ...]
    concerning_report_ids: tuple[str, ...]
    severity: Severity
    resolved: bool = False


class CouncilAudit(FrozenModel):
    approved: bool
    checks: tuple[PolicyCheck, ...]
    contradictions: tuple[Contradiction, ...]
    blocker_codes: tuple[str, ...]
    evidence_coverage: float = Field(ge=0, le=1)


class CouncilResult(FrozenModel):
    symbol: str
    as_of: datetime
    evidence_ids: tuple[str, ...]
    reports: tuple[ResearchReport, ...]
    audit: CouncilAudit


class CouncilVerdict(StrEnum):
    SUPPORT = "SUPPORT"
    CONCERN = "CONCERN"
    ABSTAIN = "ABSTAIN"


class CouncilSynthesis(FrozenModel):
    symbol: str
    as_of: datetime
    verdict: CouncilVerdict
    summary: str = Field(min_length=1, max_length=700)
    report_ids: tuple[str, ...] = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    risk_flags: tuple[str, ...] = ()
    model: str
    prompt_version: str


class OrchestratedCouncilResult(FrozenModel):
    council: CouncilResult
    synthesis: CouncilSynthesis | None
    route: str
