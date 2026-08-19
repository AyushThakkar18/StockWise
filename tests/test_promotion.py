from portfoliopilot.promotion import PromotionPolicy, evaluate_promotion


def test_all_promotion_gates_must_pass() -> None:
    metrics = {
        "independent_dates": 100, "rank_correlation": 0.05, "information_ratio": 0.3,
        "max_drawdown": -0.15, "annual_turnover": 4.0, "holdout_opened": 1,
    }
    assert all(check.passed for check in evaluate_promotion(metrics, PromotionPolicy()))
    metrics["holdout_opened"] = 0
    assert not all(check.passed for check in evaluate_promotion(metrics, PromotionPolicy()))
