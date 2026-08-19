from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .dataset import TrainingRow, date_demean_labels


@dataclass(frozen=True)
class LinearModelArtifact:
    model_version: str
    feature_version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float
    training_rows: int
    training_end: str
    artifact_hash: str

    def predict(self, features: dict[str, float]) -> float:
        missing = set(self.feature_names) - set(features)
        if missing:
            raise ValueError(f"missing model features: {sorted(missing)}")
        standardized = [
            (features[name] - mean) / scale
            for name, mean, scale in zip(self.feature_names, self.means, self.scales, strict=True)
        ]
        return self.intercept + sum(
            value * coefficient
            for value, coefficient in zip(standardized, self.coefficients, strict=True)
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> LinearModelArtifact:
        values: dict[str, Any] = json.loads(payload)
        for key in ("feature_names", "means", "scales", "coefficients"):
            values[key] = tuple(values[key])
        artifact = cls(**values)
        expected = _artifact_hash({key: value for key, value in values.items() if key != "artifact_hash"})
        if artifact.artifact_hash != expected:
            raise ValueError("model artifact hash mismatch")
        return artifact


def train_ridge(
    rows: tuple[TrainingRow, ...], alpha: float, model_version: str
) -> LinearModelArtifact:
    if not rows or alpha < 0:
        raise ValueError("training rows are required and alpha cannot be negative")
    versions = {row.feature_version for row in rows}
    if len(versions) != 1:
        raise ValueError("mixed feature versions are not allowed")
    feature_names = tuple(sorted(rows[0].features))
    if any(tuple(sorted(row.features)) != feature_names for row in rows):
        raise ValueError("inconsistent feature schema")
    matrix = np.asarray([[row.features[name] for name in feature_names] for row in rows], dtype=float)
    targets = np.asarray(date_demean_labels(rows), dtype=float)
    scaler = StandardScaler().fit(matrix)
    model = Ridge(alpha=alpha).fit(scaler.transform(matrix), targets)
    core = {
        "model_version": model_version, "feature_version": next(iter(versions)),
        "feature_names": feature_names, "means": tuple(float(value) for value in scaler.mean_),
        "scales": tuple(float(value) for value in scaler.scale_),
        "coefficients": tuple(float(value) for value in model.coef_),
        "intercept": float(model.intercept_), "alpha": alpha,
        "training_rows": len(rows), "training_end": max(row.label_end_date for row in rows).isoformat(),
    }
    return LinearModelArtifact(**core, artifact_hash=_artifact_hash(core))


def _artifact_hash(values: dict[str, Any]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(canonical.encode()).hexdigest()

