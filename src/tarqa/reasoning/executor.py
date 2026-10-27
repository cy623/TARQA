from __future__ import annotations

from pathlib import Path
from typing import Any
from threading import Timer

import duckdb

from tarqa.data.sql_utils import assert_read_only_sql


class DuckDBReadOnlyExecutor:
    def __init__(self, database_path: str | Path, *, timeout_seconds: float = 120) -> None:
        self.database_path = Path(database_path)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def execute(self, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        assert_read_only_sql(sql)
        connection = duckdb.connect(str(self.database_path), read_only=True, config={"enable_external_access": False})
        timer = Timer(self.timeout_seconds, connection.interrupt)
        timer.daemon = True
        timer.start()
        try:
            cursor = connection.execute(sql)
            columns = [item[0] for item in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return columns, rows
        finally:
            timer.cancel()
            timer.join()
            connection.close()
