from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from statistics import fmean


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    independent_dates: int
    resamples: int
    seed: int


def date_clustered_mean_interval(
    observations: dict[date, tuple[float, ...]], confidence: float = 0.95,
    resamples: int = 2_000, seed: int = 7,
) -> ConfidenceInterval:
    if not observations or not 0 < confidence < 1 or resamples < 100:
        raise ValueError("invalid clustered bootstrap inputs")
    dates = tuple(sorted(observations))
    if any(not values or any(not math.isfinite(value) for value in values) for values in observations.values()):
        raise ValueError("each decision date requires finite observations")
    cluster_means = {decision_date: fmean(observations[decision_date]) for decision_date in dates}
    estimate = fmean(cluster_means.values())
    generator = random.Random(seed)
    samples = sorted(
        fmean(cluster_means[generator.choice(dates)] for _ in dates)
        for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    lower_index = max(0, math.floor(tail * resamples))
    upper_index = min(resamples - 1, math.ceil((1 - tail) * resamples) - 1)
    return ConfidenceInterval(
        estimate, samples[lower_index], samples[upper_index], confidence,
        len(dates), resamples, seed,
    )

