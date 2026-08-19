from datetime import date, timedelta

from portfoliopilot.statistics import date_clustered_mean_interval


def test_clustered_interval_is_seeded_and_counts_dates_not_rows() -> None:
    start = date(2020, 1, 1)
    values = {
        start + timedelta(days=index): (float(index), float(index) + 0.5)
        for index in range(10)
    }
    left = date_clustered_mean_interval(values, resamples=500, seed=11)
    right = date_clustered_mean_interval(values, resamples=500, seed=11)
    assert left == right
    assert left.independent_dates == 10
    assert left.lower < left.estimate < left.upper

