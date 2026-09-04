from decimal import Decimal

from test_multi_backtest import bars

from portfoliopilot.reliable_strategy import ReliableCompositeStrategy


class FakeSEC:
    def company_facts(self, symbol, cik):
        def fact(value):
            return {"units": {"USD": [
                {"val": value, "form": "10-K", "filed": "2019-02-01", "start": "2018-01-01", "end": "2018-12-31"},
                {"val": value * 1.1, "form": "10-K", "filed": "2020-02-01", "start": "2019-01-01", "end": "2019-12-31"},
            ]}}
        return {"facts": {"us-gaap": {
            "Revenues": fact(100), "NetIncomeLoss": fact(10),
            "OperatingIncomeLoss": fact(15), "Assets": fact(200),
            "Liabilities": fact(80), "CashAndCashEquivalentsAtCarryingValue": fact(20),
        }}}

    def sector(self, symbol, cik):
        return f"sector-{cik % 5}"


def test_reliable_strategy_caps_positions_and_sectors() -> None:
    symbols = tuple(f"S{i:02d}" for i in range(15))
    history = {symbol: bars(symbol, list(range(100 + i, 400 + i))) for i, symbol in enumerate(symbols)}
    strategy = ReliableCompositeStrategy(
        bars("SPY", list(range(100, 400))), FakeSEC(),
        {symbol: index for index, symbol in enumerate(symbols)},
    )
    targets = strategy.targets(history)
    assert len(targets) == 15
    assert sum(targets.values()) <= Decimal(1)
    assert max(targets.values()) <= Decimal("0.10")
    by_sector = {}
    for symbol, weight in targets.items():
        sector = FakeSEC().sector(symbol, strategy.cik_by_symbol[symbol])
        by_sector[sector] = by_sector.get(sector, Decimal(0)) + weight
    assert max(by_sector.values()) <= Decimal("0.25")
