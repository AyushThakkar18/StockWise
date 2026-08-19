from datetime import timezone
from decimal import Decimal

from .contracts import ExecutionResult, Fill, MarketQuote, Order, OrderStatus, Side
from .ledger import PortfolioLedger

BPS = Decimal("10000")


class PaperBroker:
    def __init__(self, fee_per_order: Decimal = Decimal("0"), slippage_bps: Decimal = Decimal("2")):
        if fee_per_order < 0 or slippage_bps < 0:
            raise ValueError("cost assumptions cannot be negative")
        self.fee_per_order = fee_per_order
        self.slippage_bps = slippage_bps

    def execute(self, order: Order, quote: MarketQuote, ledger: PortfolioLedger) -> ExecutionResult:
        rejection = self._validate(order, quote)
        if rejection:
            return ExecutionResult(
                order_id=order.id, status=OrderStatus.REJECTED,
                residual_quantity=order.quantity, reason=rejection
            )
        quantity = min(order.quantity, quote.available_quantity)
        direction = Decimal("1") if order.side == Side.BUY else Decimal("-1")
        half_spread = quote.spread_bps / (Decimal("2") * BPS)
        slippage = self.slippage_bps / BPS
        price = quote.mid * (Decimal("1") + direction * (half_spread + slippage))
        spread_cost = quantity * quote.mid * half_spread
        slippage_cost = quantity * quote.mid * slippage
        fill = Fill(
            id=f"fill-{order.id}", order_id=order.id, symbol=order.symbol, side=order.side,
            quantity=quantity, price=price, spread_cost=spread_cost,
            slippage_cost=slippage_cost, fee=self.fee_per_order,
            executed_at=quote.observed_at.astimezone(timezone.utc),
        )
        try:
            self._check_portfolio_constraint(order, fill, quote, ledger)
            ledger.apply_fill(fill)
        except ValueError as exc:
            return ExecutionResult(
                order_id=order.id, status=OrderStatus.REJECTED,
                residual_quantity=order.quantity, reason=str(exc)
            )
        residual = order.quantity - quantity
        status = OrderStatus.FILLED if residual == 0 else OrderStatus.PARTIAL
        return ExecutionResult(order_id=order.id, status=status, fill=fill, residual_quantity=residual)

    @staticmethod
    def _validate(order: Order, quote: MarketQuote) -> str | None:
        if quote.symbol != order.symbol:
            return "quote symbol mismatch"
        if quote.halted:
            return "instrument halted"
        if quote.observed_at < order.earliest_execution_at or quote.available_at > quote.observed_at:
            return "quote unavailable for permitted execution session"
        if quote.available_quantity <= 0:
            return "no executable liquidity"
        return None

    @staticmethod
    def _check_portfolio_constraint(
        order: Order, fill: Fill, quote: MarketQuote, ledger: PortfolioLedger
    ) -> None:
        if order.side == Side.SELL:
            return
        marks = {symbol: quote.mid for symbol in ledger.positions}
        marks[order.symbol] = quote.mid
        equity_before = ledger.equity(marks)
        held = ledger.positions.get(order.symbol)
        target_value = ((held.quantity if held else Decimal("0")) + fill.quantity) * quote.mid
        if equity_before and target_value / equity_before > order.max_position_weight:
            raise ValueError("maximum position weight exceeded")

