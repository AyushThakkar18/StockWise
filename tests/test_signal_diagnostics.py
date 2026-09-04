from test_multi_backtest import bars

from portfoliopilot.signal_diagnostics import forward_signal_diagnostics


def test_signal_diagnostics_reports_positive_monotonic_ic() -> None:
    universe = {
        f"S{index}": bars(f"S{index}", [100 + day * (index + 1) for day in range(40)])
        for index in range(10)
    }
    decision_on = universe["S0"][10].session
    audits = {decision_on: {"ranked": [
        {"symbol": f"S{index}", "factor": float(index)} for index in range(10)
    ]}}
    result = forward_signal_diagnostics(audits, universe, horizon=21)
    assert result["factor"]["observations"] == 10
    assert result["factor"]["mean_rank_ic"] > 0
