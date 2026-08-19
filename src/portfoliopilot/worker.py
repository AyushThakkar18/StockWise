from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from .config import Settings
from .jobs import JobQueue, JobWorker
from .market_data import AlphaVantageClient
from .snapshots import ResearchStore
from .store import EventStore
from .telemetry import configure_logging
from .workflows import MarketSnapshotHandler

LOGGER = logging.getLogger("portfoliopilot.worker")


def build_worker(settings: Settings) -> JobWorker:
    settings.validate_worker()
    queue = JobQueue(settings.database_path)
    snapshot_handler = MarketSnapshotHandler(
        lambda: AlphaVantageClient(settings.alpha_vantage_api_key or ""),
        ResearchStore(settings.database_path), EventStore(settings.database_path),
        lambda: datetime.now(UTC),
    )
    return JobWorker(queue, {"MARKET_SNAPSHOT": snapshot_handler})


def main() -> None:
    configure_logging()
    worker = build_worker(Settings.from_env())
    LOGGER.info("paper-trading worker started")
    while True:
        if not worker.run_one():
            time.sleep(5)


if __name__ == "__main__":
    main()
