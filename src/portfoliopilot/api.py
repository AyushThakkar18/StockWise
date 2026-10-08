import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel

from .backtest import Backtester, BacktestResult, BuyAndHold, Cash, Momentum
from .broker import PaperBroker
from .config import Settings
from .contracts import ExecutionResult, MarketQuote, Order
from .dashboard_ui import dashboard_html
from .features import FeatureSnapshot, build_price_features
from .forward_paper import ForwardPaperEngine
from .jobs import JobQueue
from .ledger import PortfolioLedger
from .live_council import LiveCouncilDecision
from .live_policy import V5_COMPARISON_DATABASE, V5_COMPARISON_INITIAL_CASH
from .live_service import LivePaperService, YahooLivePrices
from .market_data import DailyBar
from .operations import orders_from_targets
from .optimizer import AllocationResult
from .paper_dashboard import PaperDashboard
from .paper_store import PaperTradingStore
from .security import BearerTokenMiddleware
from .store import EventStore
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
PRICE_REFRESH_LOCK = Lock()


class DashboardPriceRunner:
    """Minimal runner used by the dashboard; it cannot invoke research or the LLM council."""

    def __init__(self) -> None:
        self.prices = YahooLivePrices()


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


class FreezeCouncilRequest(BaseModel):
    decision: LiveCouncilDecision
    earliest_execution_at: datetime
    portfolio_marks: dict[str, Decimal]


class ExecuteNextOpenRequest(BaseModel):
    decision_id: str
    execution_at: datetime
    opening_prices: dict[str, Decimal]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "paper"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/dashboard", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "execution": "paper_only",
        "strategies": ["cash", "buy_and_hold", "momentum"],
        "live_broker": False,
        "llm_order_authority": False,
    }


def _dashboard(database=None) -> PaperDashboard:
    path = database or settings.database_path
    return PaperDashboard(PaperTradingStore(path), EventStore(path))


@app.get("/dashboard/overview")
def dashboard_overview() -> dict[str, object]:
    return _dashboard().overview()


@app.post("/dashboard/refresh-prices")
def dashboard_refresh_prices() -> dict[str, object]:
    if not PRICE_REFRESH_LOCK.acquire(blocking=False):
        return {"status": "REFRESH_IN_PROGRESS", "valuation_updated": False}
    try:
        result = LivePaperService(settings, DashboardPriceRunner()).price_cycle()
        return {"status": "UPDATED" if result["valuation_updated"] else "NO_UPDATE", **result}
    finally:
        PRICE_REFRESH_LOCK.release()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return dashboard_html()


@app.get("/dashboard/decisions")
def dashboard_decisions() -> tuple[dict[str, object], ...]:
    return _dashboard().decisions()


@app.get("/dashboard/trades")
def dashboard_trades() -> tuple[dict[str, object], ...]:
    return _dashboard().trades()


@app.get("/dashboard/v5/overview")
def dashboard_v5_overview() -> dict[str, object]:
    return _dashboard(V5_COMPARISON_DATABASE).overview()


@app.get("/dashboard/v5/decisions")
def dashboard_v5_decisions() -> tuple[dict[str, object], ...]:
    return _dashboard(V5_COMPARISON_DATABASE).decisions()


@app.get("/dashboard/v5/trades")
def dashboard_v5_trades() -> tuple[dict[str, object], ...]:
    return _dashboard(V5_COMPARISON_DATABASE).trades()


@app.post("/dashboard/v5/refresh-prices")
def dashboard_v5_refresh_prices() -> dict[str, object]:
    if not PRICE_REFRESH_LOCK.acquire(blocking=False):
        return {"status": "REFRESH_IN_PROGRESS", "valuation_updated": False}
    try:
        result = LivePaperService(
            settings, DashboardPriceRunner(), database=V5_COMPARISON_DATABASE,
            initial_cash=Decimal(V5_COMPARISON_INITIAL_CASH),
        ).price_cycle()
        return {"status": "UPDATED" if result["valuation_updated"] else "NO_UPDATE", **result}
    finally:
        PRICE_REFRESH_LOCK.release()


@app.get("/operations/status")
def operations_status() -> dict[str, object]:
    events = EventStore(settings.database_path).events("live-service")
    cycles = [event for event in events if event["event_type"] == "LIVE_SERVICE_CYCLE"]
    return {
        "execution_mode": "SIMULATED_NEXT_OPEN",
        "broker_required": False,
        "job_queue": JobQueue(settings.database_path).summary(),
        "latest_service_cycle": json.loads(cycles[-1]["payload"]) if cycles else None,
        "paper_portfolio": _dashboard().overview(),
    }


@app.post("/paper/council/freeze")
def freeze_council(request: FreezeCouncilRequest) -> dict[str, str]:
    try:
        payload_hash = ForwardPaperEngine(settings.database_path).freeze(
            request.decision, request.earliest_execution_at, request.portfolio_marks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"decision_id": request.decision.decision_id, "payload_hash": payload_hash}


@app.post("/paper/council/execute-next-open")
def execute_next_open(request: ExecuteNextOpenRequest) -> dict[str, object]:
    try:
        results = ForwardPaperEngine(settings.database_path).execute_next_open(
            request.decision_id, request.execution_at, request.opening_prices,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "decision_id": request.decision_id,
        "execution_mode": "SIMULATED_NEXT_OPEN",
        "results": results,
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
