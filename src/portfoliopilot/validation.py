from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class TimeFold:
    train: tuple[date, ...]
    validate: tuple[date, ...]


def purged_walk_forward(
    dates: tuple[date, ...],
    minimum_train: int,
    validation_size: int,
    purge_size: int,
    embargo_size: int,
) -> tuple[TimeFold, ...]:
    if dates != tuple(sorted(set(dates))):
        raise ValueError("dates must be sorted and unique")
    if min(minimum_train, validation_size) <= 0 or min(purge_size, embargo_size) < 0:
        raise ValueError("invalid split sizes")
    folds = []
    validation_start = minimum_train + purge_size
    while validation_start + validation_size <= len(dates):
        train_end = validation_start - purge_size
        folds.append(TimeFold(dates[:train_end], dates[validation_start:validation_start + validation_size]))
        validation_start += validation_size + embargo_size
    if not folds:
        raise ValueError("not enough dates for one fold")
    return tuple(folds)


@dataclass(frozen=True)
class ProtocolDates:
    development_start: date
    development_end: date
    validation_start: date
    validation_end: date
    holdout_start: date
    holdout_end: date

    def validate(self, embargo_days: int) -> None:
        if not (
            self.development_start <= self.development_end
            < self.validation_start <= self.validation_end
            < self.holdout_start <= self.holdout_end
        ):
            raise ValueError("protocol ranges must be ordered and disjoint")
        embargo = timedelta(days=embargo_days)
        if self.validation_start - self.development_end <= embargo:
            raise ValueError("development/validation embargo is too short")
        if self.holdout_start - self.validation_end <= embargo:
            raise ValueError("validation/holdout embargo is too short")

