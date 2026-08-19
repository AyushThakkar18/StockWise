from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import LinearModelArtifact


class ModelRegistry:
    """Immutable model artifacts; evaluations and status changes append separately."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS model_artifacts (
                model_version TEXT PRIMARY KEY,
                artifact_hash TEXT NOT NULL UNIQUE,
                artifact_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL REFERENCES model_artifacts(model_version),
                experiment_id TEXT NOT NULL,
                stage TEXT NOT NULL CHECK(stage IN ('DEVELOPMENT', 'VALIDATION', 'HOLDOUT')),
                metrics_json TEXT NOT NULL,
                eligible_for_paper INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(model_version, experiment_id, stage)
            );
            CREATE TRIGGER IF NOT EXISTS no_artifact_update
            BEFORE UPDATE ON model_artifacts
            BEGIN SELECT RAISE(ABORT, 'model artifact is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS no_evaluation_update
            BEFORE UPDATE ON model_evaluations
            BEGIN SELECT RAISE(ABORT, 'model evaluation is immutable'); END;
        """)
        self.connection.commit()

    def register(self, artifact: LinearModelArtifact) -> None:
        # Round-trip validation verifies the hash before persistence.
        LinearModelArtifact.from_json(artifact.to_json())
        self.connection.execute(
            "INSERT INTO model_artifacts VALUES (?, ?, ?, ?)",
            (artifact.model_version, artifact.artifact_hash, artifact.to_json(), datetime.now(UTC).isoformat()),
        )
        self.connection.commit()

    def record_evaluation(
        self, model_version: str, experiment_id: str, stage: str,
        metrics: dict[str, float], eligible_for_paper: bool,
    ) -> None:
        if stage != "HOLDOUT" and eligible_for_paper:
            raise ValueError("only a holdout evaluation can enable paper trading")
        self.connection.execute(
            "INSERT INTO model_evaluations(model_version, experiment_id, stage, metrics_json, "
            "eligible_for_paper, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (model_version, experiment_id, stage, json.dumps(metrics, sort_keys=True),
             int(eligible_for_paper), datetime.now(UTC).isoformat()),
        )
        self.connection.commit()

    def artifact(self, model_version: str) -> LinearModelArtifact:
        row = self.connection.execute(
            "SELECT artifact_json FROM model_artifacts WHERE model_version = ?", (model_version,)
        ).fetchone()
        if row is None:
            raise KeyError(model_version)
        return LinearModelArtifact.from_json(row["artifact_json"])

