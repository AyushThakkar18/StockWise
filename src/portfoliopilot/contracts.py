from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Quality(StrEnum):
    PASS = "PASS"
    STALE = "STALE"
    CONFLICT = "CONFLICT"
    MISSING = "MISSING"


class Action(StrEnum):
    BUY = "BUY"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    NO_TRADE = "NO_TRADE"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"


class PositionState(StrEnum):
    CASH_OR_SPY = "CASH_OR_SPY"
    CANDIDATE = "CANDIDATE"
    APPROVED = "APPROVED"
    ORDER_PENDING = "ORDER_PENDING"
    OPEN = "OPEN"
    ADD_ALLOWED = "ADD_ALLOWED"
    REDUCE_REQUIRED = "REDUCE_REQUIRED"
    EXIT_REQUIRED = "EXIT_REQUIRED"
    CLOSED = "CLOSED"
    BLOCKED_DATA = "BLOCKED_DATA"
    BLOCKED_RISK = "BLOCKED_RISK"


class Evidence(FrozenModel):
    id: str
    symbol: str
    claim: str
    source: str
    source_url: HttpUrl | None = None
    observed_at: datetime
    published_at: datetime
    available_to_strategy_at: datetime
    retrieved_at: datetime
    vintage: str
    quality: Quality


class AgentReport(FrozenModel):
    report_id: str
    role: str
    symbol: str
    as_of: datetime
    findings: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    blockers: tuple[str, ...] = ()


class Forecast(FrozenModel):
    symbol: str
    as_of: datetime
    expected_return: Decimal
    benchmark_return: Decimal
    expected_excess_after_costs: Decimal
    bear_return: Decimal
    base_return: Decimal
    bull_return: Decimal
    downside_risk: Decimal = Field(ge=0)
    dispersion: Decimal = Field(ge=0)
    evidence_completeness: Decimal = Field(ge=0, le=1)
    calibration_status: str
    model_version: str


class Decision(FrozenModel):
    id: str
    symbol: str
    decided_at: datetime
    action: Action
    current_weight: Decimal = Field(ge=0, le=1)
    target_weight: Decimal = Field(ge=0, le=1)
    reason_codes: tuple[str, ...]
    thesis: str
    invalidation_conditions: tuple[str, ...]
    next_review_date: date
    evidence_ids: tuple[str, ...]
    policy_version: str
    model_version: str
    portfolio_snapshot_hash: str


class Order(FrozenModel):
    id: str
    decision_id: str
    symbol: str
    side: Side
    quantity: Decimal = Field(gt=0)
    submitted_at: datetime
    earliest_execution_at: datetime
    reference_price: Decimal = Field(gt=0)
    max_position_weight: Decimal = Field(gt=0, le=1)
    policy_version: str

    @model_validator(mode="after")
    def execution_must_follow_submission(self) -> Order:
        if self.earliest_execution_at <= self.submitted_at:
            raise ValueError("execution must be in a later session")
        return self


class MarketQuote(FrozenModel):
    symbol: str
    observed_at: datetime
    available_at: datetime
    mid: Decimal = Field(gt=0)
    spread_bps: Decimal = Field(ge=0)
    available_quantity: Decimal = Field(ge=0)
    halted: bool = False


class Fill(FrozenModel):
    id: str
    order_id: str
    symbol: str
    side: Side
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0)
    spread_cost: Decimal = Field(ge=0)
    slippage_cost: Decimal = Field(ge=0)
    fee: Decimal = Field(ge=0)
    executed_at: datetime


class ExecutionResult(FrozenModel):
    order_id: str
    status: OrderStatus
    fill: Fill | None = None
    residual_quantity: Decimal = Field(ge=0)
    reason: str | None = None


class PolicyCheck(FrozenModel):
    name: str
    passed: bool
    detail: str


class AuditEvent(FrozenModel):
    sequence: int
    occurred_at: datetime
    event_type: str
    entity_id: str
    payload: dict[str, Any]
