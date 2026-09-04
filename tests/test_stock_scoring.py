from datetime import date

import pytest

from portfoliopilot.kronos_forecast import summarize_paths
from portfoliopilot.stock_scoring import StockAssessment, overall_score, select_diversified


def assessment() -> StockAssessment:
    return StockAssessment(
        business_quality=80, growth_outlook=80, financial_strength=80,
        catalyst_strength=80, technical_strength=80, forecast_strength=80,
        downside_resilience=80, evidence_confidence=80,
        thesis="Consistent evidence", key_risks=("Forecast uncertainty",),
    )


def test_code_computes_anchored_overall_score() -> None:
    assert overall_score(assessment()) == 80


def test_diversified_selection_sorts_scores_and_limits_sectors() -> None:
    ranked = [(f"S{index}", 100 - index, index, "Technology" if index < 10 else "Other")
              for index in range(30)]
    selected = select_diversified(ranked, count=10, maximum_per_sector=5)
    assert selected == ("S0", "S1", "S2", "S3", "S4", "S10", "S11", "S12", "S13", "S14")


def test_kronos_path_summary_is_bounded_and_interpretable() -> None:
    forecast = summarize_paths(
        [[101, 102, 103], [99, 100, 104], [98, 97, 96]], 100, "Kronos-base", date(2025, 1, 1),
    )
    assert forecast.horizon == 3
    assert forecast.paths == 3
    assert forecast.probability_positive == 2 / 3
    assert forecast.bear_return == pytest.approx(-0.04)
    assert forecast.bull_return == pytest.approx(0.04)
