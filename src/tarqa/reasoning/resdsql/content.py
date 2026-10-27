"""Question-dependent content linking from the actual read-only DuckDB snapshot."""
from functools import lru_cache

from tarqa.reasoning.executor import DuckDBReadOnlyExecutor


def quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


class DuckDBContents:
    def __init__(self, database_path, catalog, *, max_distinct=1000, max_matches=2,
                 max_value_chars=80, timeout_seconds=120):
        self.catalog = catalog
        self.executor = DuckDBReadOnlyExecutor(database_path, timeout_seconds=timeout_seconds)
        self.max_distinct, self.max_matches = int(max_distinct), int(max_matches)
        self.max_value_chars = int(max_value_chars)
        if min(self.max_distinct, self.max_matches, self.max_value_chars) < 1:
            raise ValueError("Content limits must be positive")
        self._values = lru_cache(maxsize=512)(self._read_values)

    def _read_values(self, table, column):
        dtype = self.catalog.tables[table].columns[column].dtype.upper()
        if not any(t in dtype for t in ("CHAR", "TEXT", "STRING", "ENUM")):
            return []
        col = quote_identifier(column)
        value = f"CAST({col} AS VARCHAR)"
        sql = (f"SELECT DISTINCT {value} AS content_value FROM {quote_identifier(table)} "
               f"WHERE {col} IS NOT NULL AND length({value}) BETWEEN 1 AND {self.max_value_chars} "
               f"ORDER BY content_value LIMIT {self.max_distinct}")
        return [row[0] for row in self.executor.execute(sql)[1]]

    def matches(self, question, table, column):
        from tarqa.reasoning.resdsql.bridge import get_matched_entries
        matches = get_matched_entries(question, self._values(table, column)) or []
        return list(dict.fromkeys(m[1][0] for m in matches))[:self.max_matches]

    def close(self):
        self._values.cache_clear()
