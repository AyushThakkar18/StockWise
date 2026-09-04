from datetime import date
from pathlib import Path

import pytest

from portfoliopilot.kronos_forecast import KronosForecast, KronosForecaster
from portfoliopilot.kronos_only_backtest import forecast_score


def forecast(median: float, dispersion: float, drawdown: float) -> KronosForecast:
    return KronosForecast(
        model="test", as_of=date(2025, 1, 1), horizon=21, paths=10,
        median_return=median, mean_return=median, probability_positive=.5,
        bear_return=drawdown, bull_return=median, predicted_volatility=.2,
        predicted_max_drawdown=drawdown, forecast_dispersion=dispersion,
    )


def test_forecast_score_rewards_excess_return_and_penalizes_risk() -> None:
    benchmark = forecast(.02, .01, -.05)
    assert forecast_score(forecast(.08, .01, -.05), benchmark) > forecast_score(
        forecast(.04, .08, -.20), benchmark,
    )


def test_forecaster_rejects_invalid_stability_controls() -> None:
    with pytest.raises(ValueError, match="stability controls"):
        KronosForecaster(Path("repo"), Path("cache"), recycle_every=-1)
    with pytest.raises(ValueError, match="stability controls"):
        KronosForecaster(Path("repo"), Path("cache"), cooldown_seconds=-.1)
