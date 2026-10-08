from test_multi_backtest import bars

from portfoliopilot.bounded_agent_strategy import (
    bounded_agent_adjustment,
    classify_return_paths,
    feature_packet,
)
from portfoliopilot.bynara_agents import AgentFeature


def feature(role: str, signal: int, confidence: float = 1) -> AgentFeature:
    return AgentFeature(
        role=role, signal=signal, confidence=confidence, reason="Measured feature assessment",
        risk_flags=(),
    )


def test_agent_adjustment_is_bounded_and_directional() -> None:
    positive = {
        role: feature(role, 2) for role in ("TECHNICAL", "QUALITY", "RISK")
    }
    negative = {
        role: feature(role, -2) for role in ("TECHNICAL", "QUALITY", "RISK")
    }
    assert bounded_agent_adjustment(positive) == .08
    assert bounded_agent_adjustment(negative) == -.08


def test_low_confidence_shrinks_agent_influence() -> None:
    features = {
        role: feature(role, 2, 0) for role in ("TECHNICAL", "QUALITY", "RISK")
    }
    assert bounded_agent_adjustment(features) == 0


def test_feature_packet_separates_momentum_horizons_and_return_path() -> None:
    history = bars("A", list(range(100, 700)))
    benchmark = bars("SPY", list(range(200, 800)))
    factor = {
        "rank": 1, "score": .9, "momentum_12_1": .5, "momentum_6_1": .2,
        "relative_strength": .1, "trend": .3, "growth": .1, "quality": .2,
        "low_volatility": -.2, "downside_risk": -.1, "sector": "Technology",
        "fundamental_completeness": 1.0, "dollar_volume": 10_000_000,
        "factor_percentiles": {"momentum_12_1": .8, "growth": .6},
    }

    packet = feature_packet("A", history, benchmark, factor)
    prices = packet["price_features"]

    assert set(packet["deterministic_factors"]) >= {
        "momentum_12_1", "momentum_6_1", "trend", "growth",
    }
    assert packet["factor_percentiles"] == {"momentum_12_1": .8, "growth": .6}
    assert prices["return_12_to_1_months"] == float(
        history[-22].adjusted_close / history[-253].adjusted_close - 1
    )
    assert -1 <= prices["information_discreteness_12_to_1"] <= 1
    assert 0 <= prices["positive_day_fraction_12_to_1"] <= 1
    assert prices["price_to_52_week_high"] == 1
    assert packet["market_regime"]["spy_return_504d"] is not None
    assert "PRICE_MOMENTUM" in packet["signal_provenance"]


def test_return_path_labels_are_cross_sectional_and_do_not_change_scores() -> None:
    packets = {
        f"S{index}": {
            "deterministic_score": index,
            "price_features": {"information_discreteness_12_to_1": value},
        }
        for index, value in enumerate((-.8, -.3, 0, .3, .8))
    }

    classify_return_paths(packets)

    assert packets["S0"]["price_features"]["return_path_label"] == "CONTINUOUS"
    assert packets["S2"]["price_features"]["return_path_label"] == "MIXED"
    assert packets["S4"]["price_features"]["return_path_label"] == "DISCRETE"
    assert [packets[f"S{index}"]["deterministic_score"] for index in range(5)] == list(range(5))
