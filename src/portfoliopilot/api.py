import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .backtest import Backtester, BacktestResult, BuyAndHold, Cash, Momentum
from .broker import PaperBroker
from .config import Settings
from .contracts import ExecutionResult, MarketQuote, Order
from .features import FeatureSnapshot, build_price_features
from .ledger import PortfolioLedger
from .market_data import DailyBar
from .operations import orders_from_targets
from .optimizer import AllocationResult
from .security import BearerTokenMiddleware
from .telemetry import configure_logging

app = FastAPI(
    title="PortfolioPilot AI",
    description="Deterministic paper trading only. No live brokerage connectivity.",
    version="0.1.0",
)
settings = Settings.from_env()
if settings.require_api_token:
    settings.validate_exposed_api()
configure_logging()
app.add_middleware(BearerTokenMiddleware, token=settings.api_token)
LOGGER = logging.getLogger("portfoliopilot.api")
ledger = PortfolioLedger(Decimal("100000"))
broker = PaperBroker()


@app.middleware("http")
async def request_telemetry(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    duration = round((time.perf_counter() - started) * 1000, 2)
    response.headers["x-request-id"] = request_id
    LOGGER.info(
        "request completed",
        extra={
            "request_id": request_id, "method": request.method, "path": request.url.path,
            "status_code": response.status_code, "duration_ms": duration,
        },
    )
    return response


class AccountView(BaseModel):
    cash: Decimal
    realized_pnl: Decimal
    positions: dict[str, dict[str, Decimal]]


class ExecutionRequest(BaseModel):
    order: Order
    quote: MarketQuote


class BacktestRequest(BaseModel):
    bars: tuple[DailyBar, ...]
    benchmark_bars: tuple[DailyBar, ...]
    strategy: str = "momentum"
    momentum_lookback: int = 126
    starting_cash: Decimal = Decimal(100_000)
    cost_bps: Decimal = Decimal(5)


class FeatureRequest(BaseModel):
    bars: tuple[DailyBar, ...]
    benchmark_bars: tuple[DailyBar, ...]
    decision_at: datetime


class TargetOrderRequest(BaseModel):
    session_id: str
    decision_at: datetime
    earliest_execution_at: datetime
    target_weights: dict[str, float]
    marks: dict[str, Decimal]
    maximum_position_weight: Decimal = Decimal("0.10")
    policy_version: str = "allocation-policy-v1"


class ValuationRequest(BaseModel):
    marks: dict[str, Decimal]
    benchmark_value: Decimal
    initial_portfolio_value: Decimal = Decimal(100_000)
    initial_benchmark_value: Decimal = Decimal(100_000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "paper"}


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "execution": "paper_only",
        "strategies": ["cash", "buy_and_hold", "momentum"],
        "live_broker": False,
        "llm_order_authority": False,
    }


@app.get("/account", response_model=AccountView)
def account() -> AccountView:
    return AccountView(
        cash=ledger.cash,
        realized_pnl=ledger.realized_pnl,
        positions={
            symbol: {"quantity": p.quantity, "cost_basis": p.cost_basis, "average_cost": p.average_cost}
            for symbol, p in ledger.positions.items() if p.quantity
        },
    )


@app.post("/paper/orders/execute", response_model=ExecutionResult)
def execute(request: ExecutionRequest) -> ExecutionResult:
    result = broker.execute(request.order, request.quote, ledger)
    if result.status.value == "REJECTED":
        raise HTTPException(status_code=422, detail=result.reason)
    return result


@app.post("/research/backtests", response_model=BacktestResult)
def backtest(request: BacktestRequest) -> BacktestResult:
    strategies = {
        "cash": Cash(),
        "buy_and_hold": BuyAndHold(),
        "momentum": Momentum(request.momentum_lookback),
    }
    strategy = strategies.get(request.strategy)
    if not strategy:
        raise HTTPException(status_code=422, detail="unknown baseline strategy")
    try:
        return Backtester(request.starting_cash, request.cost_bps).run(
            request.bars, request.benchmark_bars, strategy
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/research/features", response_model=FeatureSnapshot)
def features(request: FeatureRequest) -> FeatureSnapshot:
    try:
        return build_price_features(request.bars, request.benchmark_bars, request.decision_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/operations/orders", response_model=tuple[Order, ...])
def operational_orders(request: TargetOrderRequest) -> tuple[Order, ...]:
    allocation = AllocationResult(
        target_weights=request.target_weights,
        cash_weight=max(0.0, 1 - sum(request.target_weights.values())),
        checks={}, turnover=0.0,
    )
    try:
        return orders_from_targets(
            request.session_id, request.decision_at, request.earliest_execution_at,
            allocation, ledger, request.marks, request.maximum_position_weight,
            request.policy_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/operations/account/valuation")
def operational_valuation(request: ValuationRequest) -> dict[str, Decimal]:
    try:
        portfolio_value = ledger.equity(request.marks)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    gross_exposure = sum(
        (
            position.quantity * request.marks[symbol]
            for symbol, position in ledger.positions.items() if position.quantity
        ),
        Decimal(0),
    )
    return {
        "portfolio_value": portfolio_value,
        "benchmark_value": request.benchmark_value,
        "cash": ledger.cash,
        "gross_exposure": gross_exposure,
        "total_return": portfolio_value / request.initial_portfolio_value - 1,
        "benchmark_return": request.benchmark_value / request.initial_benchmark_value - 1,
        "excess_return": portfolio_value / request.initial_portfolio_value
        - request.benchmark_value / request.initial_benchmark_value,
    }
