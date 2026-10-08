from test_multi_backtest import bars

from portfoliopilot.portfolio_construction import estimate_shrunk_correlations, pairwise_summary


def test_shrunk_correlations_are_period_aligned_and_bounded() -> None:
    base = list(range(100, 260))
    histories = {
        "A": bars("A", base),
        "B": bars("B", [value * 2 for value in base]),
        "C": bars("C", [300 - value // 2 for value in base]),
    }

    estimate = estimate_shrunk_correlations(histories, ("A", "B", "C"))

    assert estimate is not None
    assert estimate.observations == 126
    assert set(estimate.correlations) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert all(-1 <= value <= 1 for value in estimate.correlations.values())
    average, maximum = pairwise_summary(("A", "B", "C"), estimate)
    assert average is not None
    assert maximum is not None
