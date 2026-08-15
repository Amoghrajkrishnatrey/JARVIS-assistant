"""
tools/sql_engine.py
DuckDB-backed query engine. Auto-registers every CSV/Parquet file in the
configured data directory as a view (named after the filename stem), and
lets the agent register or query additional files on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from ..config import settings


@dataclass
class QueryResult:
    success: bool
    columns: list[str]
    rows: list[tuple]
    row_count: int
    error: str = ""

    def preview(self, limit: int = 15) -> str:
        if not self.success:
            return f"SQL error: {self.error}"
        if not self.columns:
            return f"Query OK. {self.row_count} row(s) affected."
        header = " | ".join(self.columns)
        sep = "-" * len(header)
        body = "\n".join(" | ".join(str(v) for v in row) for row in self.rows[:limit])
        footer = f"\n... ({self.row_count} rows total)" if self.row_count > limit else ""
        return f"{header}\n{sep}\n{body}{footer}"


class DuckDBEngine:
    """Wraps a DuckDB connection, auto-registering local files as views."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or ":memory:"
        self.conn = duckdb.connect(self.db_path)
        self._register_data_dir()

    def _register_data_dir(self) -> None:
        for path in settings.data_dir.glob("*"):
            self._register_path(path, quiet=True)

    def _register_path(self, path: Path, quiet: bool = False) -> str:
        # DuckDB DDL statements (CREATE VIEW ...) can't be parameter-bound,
        # so the path is escaped and inlined rather than passed as '?'.
        stem = path.stem.replace("-", "_").replace(" ", "_")
        escaped_path = str(path).replace("'", "''")
        try:
            if path.suffix == ".csv":
                self.conn.execute(f'CREATE OR REPLACE VIEW "{stem}" AS SELECT * FROM read_csv_auto(\'{escaped_path}\')')
            elif path.suffix == ".parquet":
                self.conn.execute(f'CREATE OR REPLACE VIEW "{stem}" AS SELECT * FROM read_parquet(\'{escaped_path}\')')
            else:
                return f"Unsupported file type: {path.suffix}"
        except Exception as exc:
            if quiet:
                return ""
            return f"Failed to register {path.name}: {exc}"
        return f"Registered '{stem}' from {path.name}"

    def list_tables(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SHOW TABLES").fetchall()]

    def query(self, sql: str) -> QueryResult:
        try:
            cursor = self.conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall() if cursor.description else []
            return QueryResult(True, columns, rows, len(rows))
        except Exception as exc:
            return QueryResult(False, [], [], 0, error=str(exc))

    def register_file(self, path: str) -> str:
        """Register an arbitrary CSV/Parquet file as a queryable view."""
        p = Path(path)
        if not p.exists():
            p = settings.data_dir / path
        if not p.exists():
            return f"File not found: {path}"
        return self._register_path(p)


# Module-level singleton shared by the agent's SQL tools.
engine = DuckDBEngine()
