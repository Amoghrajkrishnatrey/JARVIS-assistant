"""
tools/code_execution.py
Isolated Python execution tool for data-analysis code, replacing the
legacy script's raw eval().

Honesty about the security model: true OS-level isolation requires a
container/VM (Docker, gVisor, firejail, etc.). This tool is defense in
depth for a *local, single-trusted-user* assistant, not a multi-tenant
sandbox:
  - runs in a separate subprocess, so a crash/hang/segfault can't take
    down the main JARVIS process
  - restricted builtins and a whitelist-based __import__, so the snippet
    cannot import os/subprocess/sys/socket/shutil/etc. or call
    eval/exec/open on its own
  - wall-clock timeout, plus CPU/memory rlimits on POSIX systems
  - stdout/stderr captured and returned; no shared state with the parent

If you plan to run untrusted code or expose this to multiple users,
swap this module for a Docker- or gVisor-based sandbox before deploying.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

_ALLOWED_IMPORTS = {
    "pandas", "numpy", "polars", "sklearn", "matplotlib", "seaborn",
    "math", "statistics", "json", "re", "datetime", "itertools", "collections",
}

_RUNNER_TEMPLATE = '''
import builtins as _b

_SAFE_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "next", "print", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "StopIteration", "ZeroDivisionError",
]
_SAFE_BUILTINS = {{name: getattr(_b, name) for name in _SAFE_NAMES if hasattr(_b, name)}}

_ALLOWED = {allowed!r}
_real_import = _b.__import__

def _guarded_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _ALLOWED:
        raise ImportError(f"Import of '{{name}}' is not permitted in the JARVIS sandbox.")
    return _real_import(name, *args, **kwargs)

_SAFE_BUILTINS["__import__"] = _guarded_import

exec(compile({code!r}, "<jarvis_sandbox>", "exec"), {{"__builtins__": _SAFE_BUILTINS}})
'''


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int


def run_python_snippet(code: str, timeout: int | None = None) -> ExecutionResult:
    """
    Execute a Python snippet in an isolated subprocess with a restricted
    builtin/import surface. Intended for pandas/numpy/polars/sklearn/
    matplotlib/seaborn data-analysis code, not general-purpose scripting.
    The snippet must print() anything it wants returned to the user.
    """
    timeout = timeout or settings.exec_timeout_seconds
    runner_src = _RUNNER_TEMPLATE.format(allowed=_ALLOWED_IMPORTS, code=code)

    with tempfile.NamedTemporaryFile("w", suffix="_jarvis_runner.py", delete=False) as f:
        f.write(runner_src)
        runner_path = Path(f.name)

    def _limit_resources():  # POSIX only; skipped on Windows via preexec_fn=None
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))  # 1 GB
        except Exception:
            pass

    try:
        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(settings.workspace_dir),
            preexec_fn=_limit_resources if sys.platform != "win32" else None,
        )
        return ExecutionResult(
            success=proc.returncode == 0,
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(False, "", f"Execution timed out after {timeout}s.", -1)
    except Exception as exc:
        return ExecutionResult(False, "", f"Sandbox failure: {exc}", -1)
    finally:
        runner_path.unlink(missing_ok=True)
