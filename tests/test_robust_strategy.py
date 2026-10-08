from decimal import Decimal

from test_multi_backtest import bars
from test_reliable_strategy import FakeSEC

from portfoliopilot.robust_strategy import (
    SIMPLE_MTG_WEIGHTS,
    RobustMultifactorStrategy,
    SimpleMomentumTrendGrowthStrategy,
    _skip_month_return,
)


def test_momentum_excludes_most_recent_month() -> None:
    history = bars("A", list(range(100, 400)))
    expected = float(history[-22].adjusted_close / history[-253].adjusted_close - 1)
    assert _skip_month_return(history, 252) == expected


def test_risk_aware_targets_are_funded_and_capped() -> None:
    symbols = tuple(f"S{i:02d}" for i in range(30))
    history = {
        symbol: bars(symbol, list(range(100 + index, 400 + index)))
        for index, symbol in enumerate(symbols)
    }
    strategy = RobustMultifactorStrategy(
        bars("SPY", list(range(100, 400))), FakeSEC(),
        {symbol: index for index, symbol in enumerate(symbols)},
    )
    targets = strategy.targets(history)
    assert len(targets) == 20
    assert Decimal(0) < sum(targets.values()) <= Decimal(1)
    assert max(targets.values()) <= Decimal("0.08")
    assert strategy.audits[max(strategy.audits)]["market_regime"] == "RISK_ON"


def test_equal_weight_allocator_never_exceeds_one() -> None:
    symbols = tuple(f"S{i:02d}" for i in range(30))
    history = {
        symbol: bars(symbol, list(range(100 + index, 400 + index)))
        for index, symbol in enumerate(symbols)
    }
    strategy = RobustMultifactorStrategy(
        bars("SPY", list(range(100, 400))), FakeSEC(),
        {symbol: index for index, symbol in enumerate(symbols)}, risk_aware=False,
    )
    assert sum(strategy.targets(history).values()) <= Decimal(1)


def test_thirty_position_allocator_never_exceeds_one() -> None:
    strategy = SimpleMomentumTrendGrowthStrategy(
        (), FakeSEC(), {}, top_n=30,
        maximum_position_weight=Decimal("0.08"), maximum_sector_weight=Decimal("0.25"),
    )
    selected = [f"S{i:02d}" for i in range(30)]
    raw = {
        symbol: {"composite_score": .75, "volatility": .20, "sector": f"sector-{i % 5}"}
        for i, symbol in enumerate(selected)
    }
    targets = strategy._risk_weights(selected, raw, Decimal(1))
    assert len(targets) == 30
    assert sum(targets.values()) <= Decimal(1)


def test_simple_strategy_has_only_four_pre_registered_signals() -> None:
    strategy = SimpleMomentumTrendGrowthStrategy((), FakeSEC(), {})
    assert strategy.factor_weights == SIMPLE_MTG_WEIGHTS
    assert strategy.risk_aware is False
    assert strategy.correlation_penalty == 0


def test_simple_strategy_audit_exposes_exact_active_factor_percentiles() -> None:
    symbols = tuple(f"S{i:02d}" for i in range(8))
    history = {
        symbol: bars(symbol, list(range(100 + index, 400 + index)))
        for index, symbol in enumerate(symbols)
    }
    strategy = SimpleMomentumTrendGrowthStrategy(
        bars("SPY", list(range(100, 400))), FakeSEC(),
        {symbol: index for index, symbol in enumerate(symbols)}, top_n=5,
        maximum_position_weight=Decimal(1), maximum_sector_weight=Decimal(1),
    )

    strategy.targets(history)
    first = strategy.audits[max(strategy.audits)]["ranked"][0]

    assert set(first["factor_percentiles"]) == set(SIMPLE_MTG_WEIGHTS)


def test_score_weighting_favors_higher_composite_score() -> None:
    strategy = SimpleMomentumTrendGrowthStrategy(
        (), FakeSEC(), {}, weighting_method="score",
        maximum_position_weight=Decimal(1), maximum_sector_weight=Decimal(1),
    )
    raw = {
        "A": {"composite_score": .80, "volatility": .20, "sector": "one"},
        "B": {"composite_score": .60, "volatility": .20, "sector": "two"},
    }
    targets = strategy._risk_weights(["A", "B"], raw, Decimal(1))
    assert targets["A"] > targets["B"]
    assert sum(targets.values()) <= Decimal(1)


def test_retention_bonus_can_keep_a_near_cutoff_holding() -> None:
    history = {
        "A": bars("A", list(range(100, 400))),
        "B": bars("B", list(range(100, 400))),
    }
    strategy = SimpleMomentumTrendGrowthStrategy(
        (), FakeSEC(), {}, retention_bonus=.02, top_n=1,
    )
    strategy.previous = ("B",)
    assert strategy._diversified_selection(["A", "B"], {"A": .60, "B": .59}, history) == ["B"]
