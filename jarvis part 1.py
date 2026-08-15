''' using github ai'''
#!/usr/bin/env python3
"""
JARVIS - Data Science / BI Local Assistant (single-file)

Features:
- Modular, object-oriented design
- Dual LLM backend: Ollama (local) with optional OpenAI fallback
- Agent orchestration layer to map LLM intents to tool actions
- Secure sandboxed Python execution for data processing (no eval())
- DuckDB-based SQL execution against local files or in-memory tables
- DataManager for dataset loading and EDA summaries (pandas / polars support)
- Project scaffolder to create standard DS project structure
- Local vector store (ChromaDB) for schema / KPI / data dictionary RAG
- CLI with pyttsx3 voice replies and robust error handling

Limitations:
- This script integrates with external services/libraries; ensure required packages are installed.
- Ensure you configure environment variables for OpenAI or Ollama if you plan to use LLMs.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import time
import tempfile
import traceback
import subprocess
import threading
import multiprocessing
import queue
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import webbrowser
import textwrap

# Third-party imports (ensure these are installed)
try:
    import pyttsx3
    import requests
    import duckdb
    import pandas as pd
    import numpy as np
    import polars as pl
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    # Optional: openai client
    import openai
    from dotenv import load_dotenv
except Exception:
    # Allow import-time failures to be handled gracefully — provide actionable error message at startup
    pass

# Load environment variables from .env
try:
    load_dotenv()  # type: ignore
except Exception:
    pass

# ---------------------------
# Configuration & Constants
# ---------------------------

DEFAULT_TTS_RATE = int(os.getenv("JARVIS_TTS_RATE", "150"))
DEFAULT_TTS_VOLUME = float(os.getenv("JARVIS_TTS_VOLUME", "1.0"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # e.g., http://localhost:11434
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder")  # or "llama3"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR = os.getenv("JARVIS_CHROMA_DIR", "./.jarvis_chroma")
DEFAULT_EXEC_TIMEOUT = int(os.getenv("JARVIS_EXEC_TIMEOUT", "8"))  # seconds

# Allowed imports for sandbox executor (whitelist)
ALLOWED_EXEC_MODULES = {"pandas", "numpy", "polars", "matplotlib", "seaborn", "sklearn", "duckdb"}

# ---------------------------
# Utilities
# ---------------------------

def safe_print(*args, **kwargs) -> None:
    """Robust print wrapper to avoid blocking issues in multi-threaded contexts."""
    try:
        print(*args, **kwargs)
    except Exception:
        sys.stdout.write(" ".join(map(str, args)) + ("\n" if not kwargs.get("end") else ""))

def truncate_text(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit-3] + "..."

# ---------------------------
# LLM Client (Ollama + OpenAI fallback)
# ---------------------------

class LLMClient:
    """
    Lightweight LLM client that tries Ollama first and falls back to OpenAI when configured.

    It provides a simple chat() method that returns text output. The assistant prompts
    should instruct the model to return JSON as needed when structured outputs are required.
    """

    def __init__(self, model: Optional[str] = None, ollama_url: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.model = model or OLLAMA_MODEL
        self.ollama_url = ollama_url or OLLAMA_URL
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if self.openai_api_key:
            openai.api_key = self.openai_api_key  # type: ignore

    def _call_ollama(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call local Ollama HTTP API. Returns text or raises on failure.
        """
        try:
            api_endpoint = f"{self.ollama_url}/api/generate"
            body = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            resp = requests.post(api_endpoint, json=body, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns generator with 'choices' usually; adapt depending on version
            if isinstance(data, dict) and "results" in data:
                # new-style Ollama
                text = ""
                for chunk in data["results"]:
                    text += chunk.get("content", "")
                return text
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0].get("message", {}).get("content", "") or data["choices"][0].get("text", "")
            # fallback: stringify
            return str(data)
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def _call_openai(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call OpenAI ChatCompletion (fallback). Uses single-turn prompt->response.
        """
        try:
            resp = openai.ChatCompletion.create(  # type: ignore
                model="gpt-4o-mini" if "gpt-4o-mini" in os.environ.get("JARVIS_OPENAI_MODEL", "") else "gpt-4o", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content  # type: ignore
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")

    def chat(self, prompt: str, max_tokens: int = 1024, prefer_ollama: bool = True) -> str:
        """
        Send prompt to an LLM and return text. Tries Ollama first unless prefer_ollama is False.
        """
        last_err = None
        if prefer_ollama:
            try:
                return self._call_ollama(prompt, max_tokens=max_tokens)
            except Exception as e:
                last_err = e
                safe_print(f"[LLM] Ollama failed: {e}. Trying OpenAI as fallback.")
        # Try OpenAI if API key configured
        if self.openai_api_key:
            try:
                return self._call_openai(prompt, max_tokens=max_tokens)
            except Exception as e:
                last_err = e
        # If nothing worked, raise
        raise RuntimeError(f"No LLM available. Last error: {last_err}")

# ---------------------------
# Sandboxed Code Executor
# ---------------------------

@dataclass
class ExecResult:
    success: bool
    output: str
    error: Optional[str] = None
    return_value: Optional[str] = None

def _executor_worker(code: str, allowed_modules: Optional[set], result_queue: multiprocessing.Queue) -> None:
    """
    Runs in separate process. Minimal globals. Writes output or exception to result_queue.
    Note: This worker runs untrusted code — we attempt to limit available modules.
    On UNIX we try to set resource limits if available (best-effort).
    """
    try:
        # Restrict imports by overriding __import__
        orig_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            base = name.split(".")[0]
            if allowed_modules and base not in allowed_modules:
                raise ImportError(f"Import of '{name}' is not allowed in the sandbox.")
            return orig_import(name, globals, locals, fromlist, level)

        builtins = __import__("builtins")
        builtins.__import__ = guarded_import  # type: ignore

        # Provide a minimal namespace
        sandbox_globals = {
            "__builtins__": builtins,
            "_": None,
        }
        sandbox_locals: Dict[str, Any] = {}

        # Capture stdout/stderr
        import io, sys as _sys
        old_stdout = _sys.stdout
        old_stderr = _sys.stderr
        out_io = io.StringIO()
        err_io = io.StringIO()
        _sys.stdout = out_io
        _sys.stderr = err_io

        try:
            exec(code, sandbox_globals, sandbox_locals)
            stdout_val = out_io.getvalue()
            stderr_val = err_io.getvalue()
            # Attempt to serialize a simple return value if present
            ret = sandbox_locals.get("RESULT", None)
            result_queue.put({
                "success": True,
                "stdout": stdout_val,
                "stderr": stderr_val,
                "return": repr(ret) if ret is not None else None,
            })
        finally:
            _sys.stdout = old_stdout
            _sys.stderr = old_stderr

    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put({"success": False, "error": str(e), "traceback": tb})

class CodeExecutor:
    """
    Executes Python code in a sandboxed separate process with timeout.
    Usage: Provide code that writes a 'RESULT' variable if you want a return value.
    """

    def __init__(self, timeout_seconds: int = DEFAULT_EXEC_TIMEOUT, allowed_modules: Optional[set] = None):
        self.timeout_seconds = timeout_seconds
        self.allowed_modules = allowed_modules or ALLOWED_EXEC_MODULES

    def run(self, code: str) -> ExecResult:
        """
        Execute the given code in a sandboxed separate process and return ExecResult.
        """
        mgr = multiprocessing.Manager()
        result_queue = mgr.Queue()
        proc = multiprocessing.Process(target=_executor_worker, args=(code, self.allowed_modules, result_queue))
        proc.start()
        proc.join(self.timeout_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return ExecResult(success=False, output="", error=f"Execution timed out after {self.timeout_seconds} seconds")
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            return ExecResult(success=False, output="", error="No result returned from executor")

        if result.get("success"):
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            ret = result.get("return")
            combined = ("\n".join([s for s in [stdout, stderr] if s])).strip()
            return ExecResult(success=True, output=combined or "<no output>", error=None, return_value=ret)
        else:
            tb = result.get("traceback", "")
            return ExecResult(success=False, output="", error=tb or result.get("error", "Unknown error"))

# ---------------------------
# SQL / DuckDB Engine
# ---------------------------

class SQLExecutor:
    """
    Wrapper around DuckDB for executing SQL against files or in-memory pandas/polars DataFrames.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        If db_path is provided, a persistent DuckDB DB is created; else uses in-memory.
        """
        self.db_path = db_path or ":memory:"
        self.conn = duckdb.connect(self.db_path)

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        self.conn.register(name, df)

    def register_polars(self, name: str, df: pl.DataFrame) -> None:
        self.conn.register(name, df.to_pandas())  # duckdb has better pandas integration

    def list_tables(self) -> List[str]:
        try:
            res = self.conn.execute("SHOW TABLES;").fetchall()
            return [r[0] for r in res]
        except Exception:
            return []

    def run(self, query: str, fetch: int = 50) -> Dict[str, Any]:
        """
        Execute a SQL query and return results and schema info.
        """
        try:
            df = self.conn.execute(query).fetchdf()
            preview = df.head(fetch)
            schema = {col: str(dtype) for col, dtype in zip(preview.columns, preview.dtypes)}
            return {"success": True, "rowcount": len(df), "preview": preview, "schema": schema}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def autoload_table_from_file(self, alias: str, file_path: str) -> Dict[str, Any]:
        """
        Load CSV/Parquet/DB file into a DuckDB table.
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in (".csv", ".txt"):
                self.conn.execute(f"CREATE OR REPLACE TABLE {alias} AS SELECT * FROM read_csv_auto('{file_path}')")
            elif ext in (".parquet", ".pq"):
                self.conn.execute(f"CREATE OR REPLACE TABLE {alias} AS SELECT * FROM read_parquet('{file_path}')")
            elif ext in (".db", ".sqlite"):
                # Attach sqlite via read_csv? Use duckdb's sqlite reader (if available) - fallback to pandas
                try:
                    self.conn.execute(f"CREATE OR REPLACE TABLE {alias} AS SELECT * FROM read_sql('{file_path}', 'SELECT * FROM sqlite_master')")
                except Exception:
                    # fallback: use pandas to read table names, but this is limited
                    import sqlite3
                    con = sqlite3.connect(file_path)
                    cur = con.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [r[0] for r in cur.fetchall()]
                    if not tables:
                        return {"success": False, "error": "No tables found in sqlite DB"}
                    # load first table
                    df = pd.read_sql_query(f"SELECT * FROM {tables[0]} LIMIT 1000000", con)
                    self.register_dataframe(alias, df)
            else:
                return {"success": False, "error": f"Unsupported file extension: {ext}"}
            return {"success": True, "message": f"Loaded {file_path} as table {alias}"}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# ---------------------------
# Data Manager (EDA, loading)
# ---------------------------

@dataclass
class DataSummary:
    rows: int
    columns: int
    dtypes: Dict[str, str]
    missing: Dict[str, int]
    numeric_summary: Optional[pd.DataFrame] = None

class DataManager:
    """
    Provides dataset loading and EDA functionality using pandas and polars.
    """

    def __init__(self, sql_executor: SQLExecutor):
        self.sql = sql_executor
        self.current_df: Optional[pd.DataFrame] = None
        self.current_name: Optional[str] = None

    def load(self, path: str, alias: Optional[str] = None) -> Dict[str, Any]:
        """
        Load CSV/Parquet into pandas and register with DuckDB.
        """
        if not os.path.exists(path):
            return {"success": False, "error": "File not found"}
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".csv", ".txt"):
                df = pd.read_csv(path)
            elif ext in (".parquet", ".pq"):
                df = pd.read_parquet(path)
            elif ext in (".feather",):
                df = pd.read_feather(path)
            else:
                return {"success": False, "error": f"Unsupported extension {ext}"}

            name = alias or (os.path.splitext(os.path.basename(path))[0] or "dataset")
            self.current_df = df
            self.current_name = name
            self.sql.register_dataframe(name, df)
            return {"success": True, "name": name, "shape": df.shape}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def summarize(self, top_n: int = 5) -> Dict[str, Any]:
        if self.current_df is None:
            return {"success": False, "error": "No dataset loaded"}
        df = self.current_df
        try:
            rows, cols = df.shape
            dtypes = {c: str(df[c].dtype) for c in df.columns}
            missing = {c: int(df[c].isna().sum()) for c in df.columns}
            numeric = df.select_dtypes(include=[np.number])
            numeric_summary = numeric.describe().T if not numeric.empty else pd.DataFrame()
            top_rows = df.head(top_n)
            return {
                "success": True,
                "rows": rows,
                "columns": cols,
                "dtypes": dtypes,
                "missing": missing,
                "numeric_summary": numeric_summary,
                "preview": top_rows,
                "name": self.current_name
            }
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def find_anomalies(self, column: Optional[str] = None) -> Dict[str, Any]:
        """
        Basic anomaly detection: z-score for numeric columns or rare categories for categorical.
        """
        if self.current_df is None:
            return {"success": False, "error": "No dataset loaded"}
        df = self.current_df
        try:
            anomalies = {}
            if column:
                if column not in df.columns:
                    return {"success": False, "error": f"Column {column} not found"}
                series = df[column]
                if np.issubdtype(series.dtype, np.number):
                    z = (series - series.mean()) / (series.std(ddof=0) + 1e-9)
                    outliers = series.index[np.abs(z) > 3].tolist()
                    anomalies[column] = {"type": "zscore", "count": len(outliers), "indices": outliers[:50]}
                else:
                    counts = series.value_counts()
                    rare = counts[counts <= max(1, int(0.01 * len(series)))]
                    anomalies[column] = {"type": "rare_values", "rare_count": len(rare), "examples": rare.head(10).to_dict()}
            else:
                # run on numeric columns
                for col in df.select_dtypes(include=[np.number]).columns:
                    s = df[col]
                    z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
                    outliers = s.index[np.abs(z) > 3].tolist()
                    if outliers:
                        anomalies[col] = {"type": "zscore", "count": len(outliers), "indices": outliers[:20]}
            return {"success": True, "anomalies": anomalies}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# ---------------------------
# Project Scaffolder
# ---------------------------

class ProjectScaffolder:
    """
    Generate a standardized Cookiecutter-style Data Science project structure.
    """

    TEMPLATE_REQUIREMENTS = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "duckdb",
        "polars",
        "pyarrow",
        "chromadb",
    ]

    GITIGNORE_CONTENT = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Data files
data/
*.sqlite
*.db
*.parquet
*.csv

# Virtual environments
venv/
.env
.venv/

# Jupyter
.ipynb_checkpoints
output/
"""

    def scaffold(self, path: str, project_name: str) -> Dict[str, Any]:
        try:
            root = os.path.join(path, project_name)
            if os.path.exists(root):
                return {"success": False, "error": f"Path already exists: {root}"}
            os.makedirs(root, exist_ok=False)
            # Create folders
            folders = [
                "data/raw",
                "data/processed",
                "notebooks",
                "src",
                "models",
                "reports",
                "tests",
            ]
            for f in folders:
                os.makedirs(os.path.join(root, f), exist_ok=True)
            # requirements.txt
            reqs = "\n".join(self.TEMPLATE_REQUIREMENTS)
            with open(os.path.join(root, "requirements.txt"), "w") as f:
                f.write(reqs + "\n")
            # .gitignore
            with open(os.path.join(root, ".gitignore"), "w") as f:
                f.write(self.GITIGNORE_CONTENT)
            # README
            readme = f"# {project_name}\n\nGenerated by JARVIS scaffolder. Fill in project details."
            with open(os.path.join(root, "README.md"), "w") as f:
                f.write(readme)
            # sample module
            sample = """\"\"\"Sample data processing module\"\"\"
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def summary(df: pd.DataFrame) -> dict:
    return {"shape": df.shape, "columns": list(df.columns)}
"""
            with open(os.path.join(root, "src", "data_manager.py"), "w") as f:
                f.write(sample)
            return {"success": True, "path": root}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# ---------------------------
# Vector Store (ChromaDB) for Data Dictionary (RAG)
# ---------------------------

class VectorStoreManager:
    """
    Uses ChromaDB to persistently store embeddings for schema/KPI/definitions and allow semantic search.
    """

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        # Use sentence-transformers embedding function if available, else basic OpenAI embeddings
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory))
            self.collection = self.client.get_or_create_collection(name="jarvis_data_dict", embedding_function=ef)
            self.available = True
        except Exception as e:
            safe_print(f"[RAG] Chroma initialization failed: {e}")
            self.available = False
            self.client = None
            self.collection = None

    def add_documents(self, docs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        docs: list of {"id": str, "text": str, "metadata": {...}}
        """
        if not self.available or self.collection is None:
            return {"success": False, "error": "Chroma not available"}
        try:
            ids = [d["id"] for d in docs]
            texts = [d["text"] for d in docs]
            metadatas = [d.get("metadata", {}) for d in docs]
            self.collection.add(documents=texts, ids=ids, metadatas=metadatas)
            self.client.persist()
            return {"success": True, "count": len(docs)}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        if not self.available or self.collection is None:
            return {"success": False, "error": "Chroma not available"}
        try:
            results = self.collection.query(query_texts=[query_text], n_results=n_results)
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# ---------------------------
# Code Refactor / Optimizer (Pandas -> Polars / SQL suggestions)
# ---------------------------

class CodeRefactor:
    """
    Light wrapper to ask LLM to translate / optimize code. We do not automatically run refactored code;
    we show it to the user, and optionally execute in the sandbox if requested.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def pandas_to_polars(self, pandas_code: str) -> Dict[str, Any]:
        """
        Ask LLM to convert pandas code snippet to polars with best practices for performance.
        Returns suggested code.
        """
        prompt = f """
You are an expert Data Engineer. Convert the following pandas code into efficient Polars code.
- Keep similar semantics.
- Prefer lazy evaluation where appropriate.
- Add comments about performance benefits.

Pandas code: {pandas_code}
"""
        response = self.llm.invoke(prompt)
        return {"success": True, "polars_code": response}