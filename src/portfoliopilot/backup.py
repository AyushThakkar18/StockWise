from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings


def backup_database(source: Path, destination_directory: Path) -> Path:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)
    destination_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_directory / f"portfoliopilot-{stamp}.db"
    if destination.exists():
        raise FileExistsError(destination)
    with sqlite3.connect(source) as source_connection, sqlite3.connect(destination) as target:
        source_connection.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError("backup integrity check failed")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and verify a PortfolioPilot SQLite backup")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    arguments = parser.parse_args()
    settings = Settings.from_env()
    output = backup_database(
        arguments.source or settings.database_path,
        arguments.destination or settings.backup_directory,
    )
    print(output)


if __name__ == "__main__":
    main()

