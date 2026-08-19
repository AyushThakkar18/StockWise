import sqlite3

from portfoliopilot.backup import backup_database


def test_backup_is_consistent_and_verified(tmp_path) -> None:
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE values_table(value TEXT)")
        connection.execute("INSERT INTO values_table VALUES ('preserved')")
        connection.commit()
    destination = backup_database(source, tmp_path / "backups")
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT value FROM values_table").fetchone()[0] == "preserved"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
