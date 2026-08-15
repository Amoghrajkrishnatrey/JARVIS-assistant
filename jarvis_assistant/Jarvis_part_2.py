''' using chatgpt '''
"""
JARVIS - Local AI Assistant for Data Science, Advanced Analytics & BI.

This module upgrades a basic CLI/TTS assistant into a tool-using local agent with:
- Ollama-first LLM orchestration and optional OpenAI-compatible cloud fallback.
- Safe-ish, subprocess-isolated Python analytics execution (no eval()).
- Read-only DuckDB querying for CSV, Parquet and local DuckDB/SQLite databases.
- EDA, project scaffolding, SQL/Pandas optimization hints, and pytest generation.
- Local ChromaDB-backed RAG memory for schemas, KPI definitions and metric formulas.
- Text CLI + pyttsx3 concise voice responses.

Design goals:
1. Keep all filesystem/data operations local by default.
2. Let the LLM select from explicit tools instead of hard-coded command matching.
3. Return actionable diagnostics instead of terminating the process on tool errors.
4. Keep provider-specific LLM details behind a small interface.

Python: 3.11+
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence

import requests
from dotenv import load_dotenv

# Optional runtime dependencies are imported lazily inside tools so the CLI can
# still start and report a useful installation error when a package is missing.

load_dotenv()

APP_NAME = "JARVIS"
APP_VERSION = "2.0.0"
DEFAULT_DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", "./.jarvis"))
DEFAULT_ALLOWED_ROOTS = [Path.cwd(), DEFAULT_DATA_DIR.resolve()]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def json_dumps(value: Any) -> str:
    """Serialize values for tool messages without failing on common objects."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def clip_text(text: str, limit: int = 12_000) -> str:
    """Keep LLM tool responses bounded so one tool cannot exhaust context."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated at {limit:,} chars]"


def require_package(module_name: str, pip_name: str | None = None) -> Any:
    """Import an optional dependency and provide an actionable install error."""
    try:
        return __import__(module_name)
    except ImportError as exc:
        package = pip_name or module_name
        raise RuntimeError(
            f"Missing optional package '{package}'. Install it with: pip install {package}"
        ) from exc


def safe_path(raw_path: str | os.PathLike[str], must_exist: bool = False) -> Path:
    """Resolve a local path and prevent obvious traversal outside approved roots."""
    candidate = Path(raw_path).expanduser().resolve()
    allowed = [p.resolve() for p in DEFAULT_ALLOWED_ROOTS]
    if not any(candidate == root or root in candidate.parents for root in allowed):
        raise PermissionError(
            f"Path '{candidate}' is outside JARVIS allowed roots. "
            f"Run JARVIS from the project directory or set JARVIS_DATA_DIR."
        )
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    return candidate


def utc_timestamp() -> str:
    """Return an ISO timestamp suitable for reports and logs."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# TTS / CLI presentation
# ---------------------------------------------------------------------------


class VoiceEngine:
    """Small pyttsx3 adapter; voice failures never terminate the assistant."""

    def __init__(self) -> None:
        self.enabled = os.getenv("JARVIS_TTS", "true").lower() not in {"0", "false", "no", "off"}
        self.rate = int(os.getenv("JARVIS_TTS_RATE", "150"))
        self.volume = float(os.getenv("JARVIS_TTS_VOLUME", "1.0"))
        self._engine: Any | None = None
        if self.enabled:
            try:
                pyttsx3 = require_package("pyttsx3")
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)
            except Exception as exc:
                print(f"[voice disabled] {exc}")
                self.enabled = False

    def speak(self, text: str) -> None:
        """Speak a concise message if TTS is enabled, while printing all text."""
        print(f"\n🤖 {APP_NAME}: {text}\n")
        if not self.enabled or self._engine is None:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            print(f"[voice warning] {exc}")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolSpec:
    """Tool definition presented to the LLM and executed by the registry."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Explicit allow-list of functions an agent may invoke."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        spec = self._tools.get(name)
        if spec is None:
            return f"ERROR: Unknown tool '{name}'. Available tools: {', '.join(self._tools)}"
        try:
            result = spec.handler(**dict(arguments))
            return clip_text(str(result))
        except Exception as exc:  # tools must never crash the main agent loop
            return (
                f"ERROR executing {name}: {type(exc).__name__}: {exc}\n"
                f"Traceback:\n{clip_text(traceback.format_exc(), 6000)}"
            )


# ---------------------------------------------------------------------------
# LLM abstraction + provider implementations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    """Normalized tool call shape shared by Ollama and OpenAI-compatible APIs."""

    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(slots=True)
class LLMResponse:
    """Normalized model response."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMBackend(Protocol):
    """Minimal interface required by the agent orchestrator."""

    @property
    def name(self) -> str: ...

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...

    def healthy(self) -> bool: ...


class OllamaBackend:
    """Ollama REST backend using its native /api/chat tool-calling protocol."""

    def __init__(self, host: str, model: str, timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def healthy(self) -> bool:
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {str(m.get("name", "")) for m in models}
            return self.model in names or any(name.startswith(self.model + ":") for name in names)
        except Exception:
            return False

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "options": {"temperature": float(os.getenv("JARVIS_TEMPERATURE", "0.1"))},
        }
        # Ollama supports a boolean or model-specific thinking configuration. We
        # intentionally disable exposing internal reasoning to the CLI.
        payload["think"] = False
        response = self._session.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(
                ToolCall(
                    name=str(fn.get("name", "")),
                    arguments=dict(args),
                    call_id=str(item.get("id") or uuid.uuid4().hex),
                )
            )
        return LLMResponse(content=str(message.get("content") or ""), tool_calls=calls, raw=data)


class OpenAICompatibleBackend:
    """Backend for OpenAI and other OpenAI-compatible cloud APIs."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return f"cloud:{self.model}"

    def healthy(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": float(os.getenv("JARVIS_TEMPERATURE", "0.1")),
        }
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(
                ToolCall(
                    name=str(fn.get("name", "")),
                    arguments=dict(args),
                    call_id=str(item.get("id") or uuid.uuid4().hex),
                )
            )
        return LLMResponse(content=str(message.get("content") or ""), tool_calls=calls, raw=data)


class FallbackLLM:
    """Ollama-first backend with optional cloud fallback."""

    def __init__(self, primary: LLMBackend | None, fallback: LLMBackend | None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_backend = "none"

    @property
    def name(self) -> str:
        return f"auto(primary={getattr(self.primary, 'name', 'none')}, fallback={getattr(self.fallback, 'name', 'none')})"

    def healthy(self) -> bool:
        return bool((self.primary and self.primary.healthy()) or (self.fallback and self.fallback.healthy()))

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        errors: list[str] = []
        if self.primary and self.primary.healthy():
            try:
                result = self.primary.chat(messages, tools)
                self.last_backend = self.primary.name
                return result
            except Exception as exc:
                errors.append(f"{self.primary.name}: {type(exc).__name__}: {exc}")
        if self.fallback and self.fallback.healthy():
            try:
                result = self.fallback.chat(messages, tools)
                self.last_backend = self.fallback.name
                return result
            except Exception as exc:
                errors.append(f"{self.fallback.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("No usable LLM backend. " + " | ".join(errors or ["Configure Ollama or cloud credentials."]))


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------


class AnalyticsEngine:
    """Dataset inspection and analytical helpers."""

    def __init__(self) -> None:
        self.default_artifacts = DEFAULT_DATA_DIR / "artifacts"
        self.default_artifacts.mkdir(parents=True, exist_ok=True)

    def _load_pandas(self, path: Path) -> Any:
        pd = require_package("pandas")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(path, lines=suffix == ".jsonl")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            duckdb = require_package("duckdb")
            table_info = duckdb.connect(str(path), read_only=True).execute("SHOW TABLES").fetchall()
            if not table_info:
                raise ValueError(f"No tables found in database: {path}")
            table = str(table_info[0][0]).replace('"', '""')
            return duckdb.connect(str(path), read_only=True).execute(f'SELECT * FROM "{table}"').df()
        raise ValueError(f"Unsupported dataset type: {suffix}")

    def eda(self, file_path: str, sample_rows: int = 8) -> str:
        """Run a local EDA pass and save a JSON report."""
        path = safe_path(file_path, must_exist=True)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        pd = require_package("pandas")
        np = require_package("numpy")
        df = self._load_pandas(path)

        numeric = df.select_dtypes(include="number")
        missing = df.isna().sum().sort_values(ascending=False)
        duplicate_rows = int(df.duplicated().sum())
        constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]

        outliers: dict[str, int] = {}
        if not numeric.empty:
            for col in numeric.columns:
                series = numeric[col].dropna()
                if series.empty:
                    continue
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                if float(iqr) == 0:
                    outliers[str(col)] = 0
                else:
                    outliers[str(col)] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

        summary = {
            "generated_at": utc_timestamp(),
            "file": str(path),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "missing": {str(k): int(v) for k, v in missing.items() if int(v) > 0},
            "missing_percent": {
                str(k): round(float(v / max(len(df), 1) * 100), 2)
                for k, v in missing.items()
                if int(v) > 0
            },
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_cols,
            "numeric_outlier_counts_iqr": outliers,
            "numeric_summary": json.loads(numeric.describe().round(4).to_json()) if not numeric.empty else {},
            "categorical_cardinality": {
                str(col): int(df[col].nunique(dropna=True))
                for col in df.select_dtypes(exclude="number").columns
            },
            "preview": df.head(max(1, min(sample_rows, 20))).to_dict(orient="records"),
        }
        out_file = self.default_artifacts / f"eda_{path.stem}_{int(time.time())}.json"
        out_file.write_text(json_dumps(summary), encoding="utf-8")

        lines = [
            f"Dataset: {path}",
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns",
            f"Duplicate rows: {duplicate_rows:,}",
            f"Constant columns: {', '.join(constant_cols) if constant_cols else 'none'}",
            "Missing values: " + (json_dumps(summary["missing"]) if summary["missing"] else "none"),
            "IQR outlier counts: " + (json_dumps(outliers) if outliers else "none"),
            "Preview:",
            df.head(max(1, min(sample_rows, 20))).to_string(index=False),
            f"Report saved to: {out_file}",
        ]
        return "\n".join(lines)

    def run_python(self, code: str, timeout_seconds: int = 30) -> str:
        """Execute approved analytics code in a separate Python process."""
        PythonSandbox.validate(code)
        artifact_dir = self.default_artifacts / f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        wrapper = textwrap.dedent(
            f"""
            import os
            os.environ.setdefault('MPLBACKEND', 'Agg')
            os.environ['JARVIS_ARTIFACT_DIR'] = r'''{artifact_dir}'''
            {code}
            """
        )
        with tempfile.TemporaryDirectory(prefix="jarvis_exec_") as tmp:
            script = Path(tmp) / "main.py"
            script.write_text(wrapper, encoding="utf-8")
            env = {
                "PATH": os.getenv("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "MPLBACKEND": "Agg",
                "JARVIS_ARTIFACT_DIR": str(artifact_dir),
            }
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=max(1, min(timeout_seconds, 120)),
            )
        payload = {
            "return_code": proc.returncode,
            "stdout": clip_text(proc.stdout, 12_000),
            "stderr": clip_text(proc.stderr, 12_000),
            "artifacts": [str(p) for p in sorted(artifact_dir.rglob("*")) if p.is_file()],
        }
        return json_dumps(payload)


class PythonSandbox:
    """AST guardrails for analytics subprocesses; this is not a full OS sandbox."""

    ALLOWED_IMPORTS = {
        "pandas",
        "polars",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "math",
        "statistics",
        "datetime",
        "typing",
        "json",
        "re",
        "pathlib",
    }
    FORBIDDEN_NAMES = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "breakpoint",
        "input",
        "help",
    }
    FORBIDDEN_ATTR_CALLS = {
        ("os", "system"),
        ("os", "popen"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("shutil", "rmtree"),
        ("shutil", "copytree"),
    }

    @classmethod
    def validate(cls, code: str) -> None:
        """Reject obvious code-execution and network primitives before subprocess execution."""
        tree = ast.parse(code, mode="exec")
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in cls.ALLOWED_IMPORTS:
                        raise PermissionError(
                            f"Import '{alias.name}' is not allowed. Allowed roots: {sorted(cls.ALLOWED_IMPORTS)}"
                        )
                    aliases[alias.asname or root] = root
            elif isinstance(node, ast.ImportFrom):
                root = node.module.split(".")[0] if node.module else ""
                if root not in cls.ALLOWED_IMPORTS:
                    raise PermissionError(f"Import from '{node.module}' is not allowed.")
                aliases[node.module.split(".")[0]] = root
            elif isinstance(node, ast.Name) and node.id in cls.FORBIDDEN_NAMES:
                raise PermissionError(f"Call to '{node.id}' is not allowed in analytics code.")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    root = aliases.get(node.func.value.id, node.func.value.id)
                    if (root, node.func.attr) in cls.FORBIDDEN_ATTR_CALLS:
                        raise PermissionError(f"Call '{root}.{node.func.attr}' is not allowed.")


# ---------------------------------------------------------------------------
# DuckDB engine
# ---------------------------------------------------------------------------


class DuckDBEngine:
    """Read-only local SQL engine for DuckDB/SQLite files and flat files."""

    READ_ONLY_PREFIXES = (
        "select",
        "with",
        "describe",
        "explain",
        "show",
        "summarize",
        "pragma",
    )

    def __init__(self) -> None:
        self.db_file = DEFAULT_DATA_DIR / "jarvis.duckdb"
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_read_only(sql: str) -> bool:
        normalized = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S).strip().lower()
        if not normalized:
            return False
        if not normalized.startswith(DuckDBEngine.READ_ONLY_PREFIXES):
            return False
        # Prevent mutating statements hidden in a CTE or multi-statement payload.
        forbidden = re.search(r"\b(insert|update|delete|merge|create|drop|alter|truncate|copy|install|load|attach|detach|export|import)\b", normalized)
        return forbidden is None

    def query(self, sql: str, database_path: str = "", limit: int = 50) -> str:
        """Execute read-only SQL and return a Markdown-like preview table."""
        duckdb = require_package("duckdb")
        if not self._is_read_only(sql):
            raise PermissionError(
                "JARVIS SQL is read-only. Allowed statements begin with SELECT, WITH, DESCRIBE, EXPLAIN, SHOW, SUMMARIZE, or PRAGMA."
            )
        sql = sql.strip().rstrip(";")
        if database_path:
            db_path = safe_path(database_path, must_exist=True)
            suffix = db_path.suffix.lower()
            if suffix == ".duckdb":
                con = duckdb.connect(str(db_path), read_only=True)
            elif suffix in {".db", ".sqlite", ".sqlite3"}:
                # SQLite files are handled read-only through sqlite3; flat files and
                # DuckDB databases are handled by DuckDB below.
                sqlite_con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
                result = sqlite_con.execute(sql)
                columns = [desc[0] for desc in (result.description or [])]
                rows = result.fetchmany(max(1, min(limit, 500)))
                sqlite_con.close()
                if not columns:
                    return "Query completed with no tabular result."
                lines = [" | ".join(str(c) for c in columns), " | ".join("---" for _ in columns)]
                lines.extend(" | ".join(str(v) for v in row) for row in rows)
                return "\n".join(lines)
            elif suffix in {".csv", ".parquet", ".pq"}:
                con = duckdb.connect()
                escaped = str(db_path).replace("'", "''")
                if suffix == ".csv":
                    con.execute(f"CREATE OR REPLACE VIEW source_file AS SELECT * FROM read_csv_auto('{escaped}')")
                else:
                    con.execute(f"CREATE OR REPLACE VIEW source_file AS SELECT * FROM read_parquet('{escaped}')")
            else:
                raise ValueError("database_path must be .csv, .parquet, .duckdb, .db, .sqlite, or .sqlite3")
        else:
            con = duckdb.connect()

        try:
            # Limit only simple SELECT/SHOW-like statements without a final LIMIT.
            executed_sql = sql
            if re.match(r"^(select|with)\b", sql, flags=re.I) and not re.search(r"\blimit\b", sql, flags=re.I):
                executed_sql = f"SELECT * FROM ({sql}) AS jarvis_subquery LIMIT {max(1, min(limit, 500))}"
            result = con.execute(executed_sql)
            columns = [desc[0] for desc in (result.description or [])]
            rows = result.fetchmany(max(1, min(limit, 500)))
            if not columns:
                return "Query completed with no tabular result."
            lines = [" | ".join(str(c) for c in columns), " | ".join("---" for _ in columns)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            if len(rows) >= limit:
                lines.append(f"\nPreview capped at {limit} rows.")
            return "\n".join(lines)
        finally:
            con.close()

    def inspect_file(self, file_path: str) -> str:
        """Return schema/row-count information without requiring a hand-written query."""
        path = safe_path(file_path, must_exist=True)
        duckdb = require_package("duckdb")
        con = duckdb.connect()
        try:
            escaped = str(path).replace("'", "''")
            suffix = path.suffix.lower()
            if suffix == ".csv":
                source = f"read_csv_auto('{escaped}')"
            elif suffix in {".parquet", ".pq"}:
                source = f"read_parquet('{escaped}')"
            elif suffix == ".duckdb":
                con.close()
                con = duckdb.connect(str(path), read_only=True)
                tables = con.execute("SHOW TABLES").fetchall()
                return json_dumps({"file": str(path), "engine": "duckdb", "tables": [row[0] for row in tables]})
            elif suffix in {".db", ".sqlite", ".sqlite3"}:
                sqlite_con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
                rows = sqlite_con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
                sqlite_con.close()
                return json_dumps({"file": str(path), "engine": "sqlite", "tables": [row[0] for row in rows]})
            else:
                raise ValueError("Unsupported file type")
            describe = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
            count = con.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
            return json_dumps({"file": str(path), "row_count": count, "columns": [list(row) for row in describe]})
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Project scaffolding and code/SQL utilities
# ---------------------------------------------------------------------------


class ProjectScaffolder:
    """Create a repeatable Data Science/BI project skeleton."""

    REQUIREMENTS = """pandas>=2.2\npolars>=1.0\nnumpy>=2.0\nscikit-learn>=1.5\nscipy>=1.13\nmatplotlib>=3.9\nseaborn>=0.13\nduckdb>=1.1\npyarrow>=17.0\npytest>=8.0\npython-dotenv>=1.0\nrequests>=2.32\nchromadb>=0.5\npyttsx3>=2.98\n"""

    GITIGNORE = """.venv/\n__pycache__/\n*.pyc\n.env\n.chroma/\n.jarvis/\n.ipynb_checkpoints/\ndata/raw/*\ndata/processed/*\nmodels/*\nreports/*\n!data/raw/.gitkeep\n!data/processed/.gitkeep\n!models/.gitkeep\n!reports/.gitkeep\n"""

    def scaffold(self, project_name: str, base_dir: str = ".") -> str:
        """Create a Cookiecutter Data Science-inspired local project template."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", project_name):
            raise ValueError("project_name must contain only letters, digits, '-', and '_'.")
        root = safe_path(base_dir, must_exist=True) / project_name
        root.mkdir(parents=True, exist_ok=False)
        dirs = [
            "data/raw",
            "data/interim",
            "data/processed",
            "notebooks",
            "src",
            "src/data",
            "src/features",
            "src/models",
            "src/utils",
            "models",
            "reports/figures",
            "tests",
            "configs",
        ]
        for rel in dirs:
            (root / rel).mkdir(parents=True, exist_ok=True)
        for rel in ["data/raw/.gitkeep", "data/processed/.gitkeep", "models/.gitkeep", "reports/figures/.gitkeep"]:
            (root / rel).touch()

        (root / "requirements.txt").write_text(self.REQUIREMENTS, encoding="utf-8")
        (root / ".gitignore").write_text(self.GITIGNORE, encoding="utf-8")
        (root / "README.md").write_text(
            f"# {project_name}\n\nGenerated by JARVIS.\n\n## Structure\n- data/raw: source datasets\n- data/processed: curated datasets\n- notebooks: exploratory work\n- src: reusable Python code\n- models: trained artifacts\n- reports: analysis outputs\n- tests: pytest suite\n",
            encoding="utf-8",
        )
        (root / "src/__init__.py").touch()
        (root / "tests/test_smoke.py").write_text(
            "def test_project_structure():\n    assert True\n",
            encoding="utf-8",
        )
        return f"Created DS/BI project at {root}\n" + "\n".join(f"- {d}/" for d in dirs) + "\n- requirements.txt\n- .gitignore\n- README.md"


class CodeOptimizer:
    """Static, deterministic optimization hints for Pandas/SQL plus pytest skeletons."""

    @staticmethod
    def optimize_pandas(code: str, target: Literal["polars", "duckdb", "pandas"] = "polars") -> str:
        suggestions: list[str] = []
        if ".apply(" in code:
            suggestions.append("Replace row-wise DataFrame.apply with vectorized expressions, Polars expressions, or DuckDB SQL.")
        if ".iterrows(" in code:
            suggestions.append("Avoid iterrows(); use vectorized operations, merge/join, groupby, or Polars expressions.")
        if "df[" in code and ".copy()" in code:
            suggestions.append("Use column projection early and copy only when mutation is required.")
        if "read_csv(" in code and "usecols" not in code:
            suggestions.append("Pass usecols= to reduce IO and memory when only a subset of columns is needed.")
        if ".astype(" in code:
            suggestions.append("Consider categorical dtypes for low-cardinality strings and nullable dtypes where appropriate.")
        if "sort_values" in code and "groupby" in code:
            suggestions.append("Check whether a full sort is necessary; groupby/aggregation may avoid an O(n log n) sort.")
        if not suggestions:
            suggestions.append("No obvious anti-patterns detected by the static rules; benchmark before and after any rewrite.")
        if target == "polars":
            suggestions.append("Polars target: push projections/filters into the lazy query plan and call collect() once.")
        elif target == "duckdb":
            suggestions.append("DuckDB target: express filtering/projection/aggregation in SQL and query Parquet directly where practical.")
        return "\n".join(f"- {s}" for s in suggestions)

    @staticmethod
    def optimize_sql(sql: str) -> str:
        suggestions: list[str] = []
        if re.search(r"select\s+\*", sql, flags=re.I):
            suggestions.append("Avoid SELECT *; project only the columns required by downstream consumers.")
        if re.search(r"\bjoin\b", sql, flags=re.I) and not re.search(r"\bon\b", sql, flags=re.I):
            suggestions.append("A JOIN without an ON predicate may be a Cartesian product; verify intentionally.")
        if re.search(r"\blike\s+'%", sql, flags=re.I):
            suggestions.append("Leading-wildcard LIKE usually prevents efficient index seeks in row stores; consider search indexes or normalized columns.")
        if re.search(r"\bwhere\s+date\s*\(", sql, flags=re.I):
            suggestions.append("Wrapping a timestamp column in DATE() can reduce predicate pushdown/index use; prefer range predicates.")
        if len(re.findall(r"\bjoin\b", sql, flags=re.I)) >= 4:
            suggestions.append("Several joins detected; verify join order, cardinality, and whether intermediate joins can be eliminated.")
        suggestions.append("For DuckDB/Parquet workloads, prioritize column projection, filter pushdown, and avoiding unnecessary materialization.")
        return "\n".join(f"- {s}" for s in suggestions)

    @staticmethod
    def generate_pytest(code: str) -> str:
        tree = ast.parse(code, mode="exec")
        functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef) and not n.name.startswith("_")]
        imports = [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
        lines = ['\"\"\"Generated by JARVIS; review before committing.\"\"\"', "import pytest", ""]
        if functions:
            for fn in functions:
                lines.extend([
                    f"def test_{fn}_smoke():",
                    f"    # TODO: add representative inputs and assertions for {fn}.",
                    "    assert callable(__import__(__name__, fromlist=['x']))",
                    "",
                ])
        if classes:
            lines.append("# Detected classes: " + ", ".join(classes))
        if not functions and not classes:
            lines.extend(["def test_placeholder():", "    assert True", ""])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local RAG memory with ChromaDB
# ---------------------------------------------------------------------------


class LocalRAGMemory:
    """Persistent local ChromaDB memory for schemas, KPIs, and business definitions."""

    def __init__(self, persist_dir: str | os.PathLike[str] | None = None) -> None:
        self.persist_dir = Path(persist_dir or (DEFAULT_DATA_DIR / "chroma"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection: Any | None = None

    def _collection_obj(self) -> Any:
        if self._collection is not None:
            return self._collection
        chromadb = require_package("chromadb")
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = client.get_or_create_collection(name="jarvis_business_memory")
        return self._collection

    @staticmethod
    def _chunk(text: str, chunk_size: int = 1400, overlap: int = 180) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size // 5)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start = max(end - overlap // 5, start + 1)
        return chunks

    def index_file(self, file_path: str) -> str:
        path = safe_path(file_path, must_exist=True)
        if path.suffix.lower() not in {".md", ".txt", ".sql", ".yaml", ".yml", ".json", ".csv"}:
            raise ValueError("RAG supports .md, .txt, .sql, .yaml, .yml, .json, and .csv files.")
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = self._chunk(text)
        collection = self._collection_obj()
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"{path}:{idx}:{chunk}".encode("utf-8")).hexdigest()
            ids.append(doc_id)
            docs.append(chunk)
            metas.append({"source": str(path), "chunk": idx, "indexed_at": utc_timestamp()})
        if ids:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return f"Indexed {len(ids)} chunks from {path} into local Chroma memory at {self.persist_dir}."

    def search(self, query: str, top_k: int = 5) -> str:
        collection = self._collection_obj()
        result = collection.query(query_texts=[query], n_results=max(1, min(top_k, 10)))
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not documents:
            return "No matching business-memory documents found."
        rows = []
        for doc, meta in zip(documents, metadatas):
            rows.append({"source": meta.get("source") if meta else None, "chunk": meta.get("chunk") if meta else None, "content": doc})
        return json_dumps(rows)


# ---------------------------------------------------------------------------
# Tool-oriented JARVIS agent
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are JARVIS, a local-first expert companion for Data Science, Advanced Analytics, and Business Intelligence.

Behavior:
- Prefer explicit tools for data access, SQL, Python analytics, EDA, scaffolding, code optimization, tests, and business-memory retrieval.
- Never claim to have executed code or SQL unless a tool result confirms it.
- For data questions, inspect available files/schema before inventing columns or metrics.
- Use local business-memory retrieval for KPI definitions, metric formulas, data dictionaries, and schemas when relevant.
- Return concise, executive-friendly conclusions followed by technical details when useful.
- When a tool fails, diagnose the error and propose a concrete correction rather than hiding it.
- Do not expose hidden chain-of-thought. Summarize reasoning as decisions, checks, and evidence instead.
- Local files are authoritative for local analysis; do not fabricate external data.
"""


class JARVISAgent:
    """Agentic orchestration loop: model -> tool calls -> tool results -> final answer."""

    def __init__(self, llm: FallbackLLM, registry: ToolRegistry, max_steps: int = 8) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max(1, max_steps)
        self.history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, user_text: str) -> str:
        """Process one user turn using an iterative tool-calling loop."""
        self.history.append({"role": "user", "content": user_text})
        for _ in range(self.max_steps):
            try:
                response = self.llm.chat(self.history, self.registry.schemas())
            except Exception as exc:
                self.history.append({"role": "assistant", "content": f"LLM backend error: {exc}"})
                return f"I couldn't reach an LLM backend. {type(exc).__name__}: {exc}"

            if response.tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in response.tool_calls
                    ],
                }
                self.history.append(assistant_message)
                for tc in response.tool_calls:
                    result = self.registry.execute(tc.name, tc.arguments)
                    # Both OpenAI-compatible APIs and current Ollama tool loops accept
                    # the following role/name structure. The provider normalizes as needed.
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.call_id,
                            "tool_name": tc.name,
                            "content": result,
                        }
                    )
                continue

            final = response.content.strip() or "I completed the requested operation but the model returned no narrative response."
            self.history.append({"role": "assistant", "content": final})
            return final
        return "I reached the maximum tool-call steps for this turn. Review the latest tool output and refine the request."

    def reset(self) -> None:
        """Clear conversation history while preserving system instructions."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def build_llm() -> FallbackLLM:
    """Construct an Ollama-first + optional cloud backend from environment variables."""
    ollama = OllamaBackend(
        host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest"),
    )

    api_key = os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("CLOUD_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    cloud_model = os.getenv("CLOUD_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    cloud = OpenAICompatibleBackend(base_url=base_url, api_key=api_key, model=cloud_model) if api_key else None

    provider = os.getenv("JARVIS_LLM_PROVIDER", "auto").lower()
    if provider == "ollama":
        return FallbackLLM(ollama, None)
    if provider in {"cloud", "openai"}:
        return FallbackLLM(None, cloud)
    return FallbackLLM(ollama, cloud)


# ---------------------------------------------------------------------------
# CLI application
# ---------------------------------------------------------------------------


class JARVISApplication:
    """Composition root: wires the engines, tools, voice, and CLI together."""

    def __init__(self) -> None:
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.voice = VoiceEngine()
        self.analytics = AnalyticsEngine()
        self.duckdb = DuckDBEngine()
        self.scaffolder = ProjectScaffolder()
        self.optimizer = CodeOptimizer()
        self.memory = LocalRAGMemory()
        self.registry = ToolRegistry()
        self._register_tools()
        self.llm = build_llm()
        self.agent = JARVISAgent(self.llm, self.registry)
        self.running = True

    def _register_tools(self) -> None:
        self.registry.register(
            ToolSpec(
                name="get_time_date",
                description="Return the current local date and time.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=lambda: dt.datetime.now().strftime("%A, %B %d, %Y %I:%M %p"),
            )
        )
        self.registry.register(
            ToolSpec(
                name="run_python_analytics",
                description=(
                    "Run approved pandas/polars/numpy/scikit-learn/matplotlib/seaborn analytics code "
                    "in a separate Python process. Do not use eval/exec or network/system calls. "
                    "Set variables/print outputs and save plots to JARVIS_ARTIFACT_DIR when useful."
                ),
                parameters={
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {"type": "string", "description": "Python analytics code."},
                        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
                    },
                },
                handler=self.analytics.run_python,
            )
        )
        self.registry.register(
            ToolSpec(
                name="eda_dataset",
                description="Perform automated EDA on a local CSV, Parquet, JSON, Excel, SQLite, or DuckDB file and save a JSON report.",
                parameters={
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {"type": "string"},
                        "sample_rows": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                    },
                },
                handler=self.analytics.eda,
            )
        )
        self.registry.register(
            ToolSpec(
                name="query_duckdb",
                description="Run a read-only SQL query against local CSV/Parquet/DuckDB/SQLite data or an in-memory DuckDB session and return a preview.",
                parameters={
                    "type": "object",
                    "required": ["sql"],
                    "properties": {
                        "sql": {"type": "string"},
                        "database_path": {"type": "string", "default": ""},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
                    },
                },
                handler=self.duckdb.query,
            )
        )
        self.registry.register(
            ToolSpec(
                name="inspect_data_file",
                description="Inspect a local dataset or database schema and row/table metadata.",
                parameters={
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
                handler=self.duckdb.inspect_file,
            )
        )
        self.registry.register(
            ToolSpec(
                name="scaffold_ds_project",
                description="Create a standardized Cookiecutter Data Science-inspired project structure with requirements, README, tests, and .gitignore.",
                parameters={
                    "type": "object",
                    "required": ["project_name"],
                    "properties": {
                        "project_name": {"type": "string"},
                        "base_dir": {"type": "string", "default": "."},
                    },
                },
                handler=self.scaffolder.scaffold,
            )
        )
        self.registry.register(
            ToolSpec(
                name="optimize_pandas_code",
                description="Inspect Pandas code for common performance anti-patterns and provide Polars/DuckDB refactoring guidance.",
                parameters={
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {"type": "string"},
                        "target": {"type": "string", "enum": ["polars", "duckdb", "pandas"], "default": "polars"},
                    },
                },
                handler=self.optimizer.optimize_pandas,
            )
        )
        self.registry.register(
            ToolSpec(
                name="optimize_sql",
                description="Analyze SQL for obvious performance and maintainability problems and explain DuckDB versus row-store considerations.",
                parameters={
                    "type": "object",
                    "required": ["sql"],
                    "properties": {"sql": {"type": "string"}},
                },
                handler=self.optimizer.optimize_sql,
            )
        )
        self.registry.register(
            ToolSpec(
                name="generate_pytest_tests",
                description="Generate a pytest starter file from Python source by inspecting its public functions/classes.",
                parameters={
                    "type": "object",
                    "required": ["code"],
                    "properties": {"code": {"type": "string"}},
                },
                handler=self.optimizer.generate_pytest,
            )
        )
        self.registry.register(
            ToolSpec(
                name="index_business_memory",
                description="Index a local schema/KPI/metric definition file into persistent local ChromaDB memory.",
                parameters={
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
                handler=self.memory.index_file,
            )
        )
        self.registry.register(
            ToolSpec(
                name="search_business_memory",
                description="Search local ChromaDB memory for company schemas, KPI definitions, metric formulas, or business glossary content.",
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                },
                handler=self.memory.search,
            )
        )

    def speak(self, message: str) -> None:
        """Print and voice a concise response."""
        # Keep speech reasonably short while preserving full text in the console.
        self.voice.speak(message)

    def help(self) -> None:
        """Display explicit slash commands; natural language remains agent-driven."""
        print(
            """
JARVIS commands
---------------
/help                 Show this help
/status               Show LLM/tooling status
/reset                Clear conversation memory
/eda <file>            Run local EDA
/sql <query>           Run read-only DuckDB SQL
/query <file> <sql>   Run SQL against a local CSV/Parquet/DB file
/index <file>          Index business glossary/schema/KPI file into local RAG
/search <query>        Search local business memory
/project <name>        Scaffold a DS/BI project
/quit                  Exit

You can also just ask JARVIS natural-language requests such as:
  "Profile sales.parquet and identify anomalous revenue values."
  "Query customers.csv for the top 10 states by revenue."
  "Refactor this Pandas code into Polars."
"""
        )

    def status(self) -> None:
        print(
            json_dumps(
                {
                    "version": APP_VERSION,
                    "llm": self.llm.name,
                    "active_backend": self.llm.last_backend,
                    "llm_healthy": self.llm.healthy(),
                    "tools": list(self.registry._tools),
                    "jarvis_data_dir": str(DEFAULT_DATA_DIR.resolve()),
                    "tts": self.voice.enabled,
                }
            )
        )

    def process_input(self, text: str) -> None:
        """Handle deterministic convenience commands or send everything else to the agent."""
        stripped = text.strip()
        lowered = stripped.lower()
        if lowered in {"/quit", "/exit", "quit", "exit", "goodbye"}:
            self.running = False
            self.speak("Goodbye. Your local workspace is unchanged unless you explicitly requested a tool action.")
            return
        if lowered in {"/help", "help", "commands"}:
            self.help()
            return
        if lowered == "/status":
            self.status()
            return
        if lowered == "/reset":
            self.agent.reset()
            self.speak("Conversation context reset.")
            return
        try:
            if lowered.startswith("/eda "):
                result = self.registry.execute("eda_dataset", {"file_path": stripped[5:].strip()})
                print(result)
                self.speak("EDA completed. The detailed report is saved under the JARVIS artifacts directory.")
                return
            if lowered.startswith("/sql "):
                result = self.registry.execute("query_duckdb", {"sql": stripped[5:].strip()})
                print(result)
                self.speak("SQL executed successfully." if not result.startswith("ERROR") else "The SQL tool returned an error; see the diagnostics above.")
                return
            if lowered.startswith("/index "):
                result = self.registry.execute("index_business_memory", {"file_path": stripped[7:].strip()})
                print(result)
                self.speak("Business-memory index updated.")
                return
            if lowered.startswith("/search "):
                result = self.registry.execute("search_business_memory", {"query": stripped[8:].strip()})
                print(result)
                self.speak("Business-memory search completed.")
                return
            if lowered.startswith("/project "):
                result = self.registry.execute("scaffold_ds_project", {"project_name": stripped[9:].strip()})
                print(result)
                self.speak("Project scaffold created.")
                return

            response = self.agent.ask(stripped)
            self.speak(response)
        except Exception as exc:
            self.speak(f"I hit an application error: {type(exc).__name__}: {exc}")

    def run(self) -> None:
        """Start the interactive CLI loop."""
        print("\n" + "=" * 76)
        print(f"🤖 {APP_NAME} {APP_VERSION} — LOCAL DS/BI AI ASSISTANT")
        print("=" * 76)
        print("Local-first agentic analytics with Ollama, DuckDB, Python tools, and RAG.")
        print("Type /help for deterministic commands, or ask naturally in plain English.\n")

        if self.llm.healthy():
            self.speak(f"JARVIS online. Active backend: {self.llm.name}.")
        else:
            self.speak("JARVIS is online, but no LLM backend is currently reachable. Check Ollama or cloud environment settings.")

        while self.running:
            try:
                command = input("👤 You: ").strip()
                if command:
                    self.process_input(command)
            except KeyboardInterrupt:
                print("\n")
                self.running = False
                self.speak("Session interrupted. Goodbye.")
            except EOFError:
                self.running = False
                print()


def main() -> int:
    """Application entry point."""
    try:
        app = JARVISApplication()
        app.run()
        return 0
    except Exception as exc:
        print(f"JARVIS failed to start: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
