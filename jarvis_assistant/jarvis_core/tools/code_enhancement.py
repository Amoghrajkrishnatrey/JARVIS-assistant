"""
tools/code_enhancement.py
LLM-driven code-quality tools: Pandas -> Polars/DuckDB refactoring, SQL
optimization review, and pytest test generation.

These are implemented as focused prompt templates around the assistant's
already-configured LLM (local Ollama or cloud fallback) rather than a
fixed rule engine, since idiomatic refactors are context-dependent. Each
`make_*_tool` factory closes over the resolved LLM so the returned
function can be registered directly as a LangChain tool.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

_REFACTOR_SYSTEM = (
    "You are a senior data engineer. Rewrite the given pandas code so it is "
    "functionally equivalent but faster, preferring Polars (lazy API) or "
    "DuckDB SQL where that is clearly more efficient for the operation shown. "
    "Briefly explain what changed and why, then give the final code in a "
    "single fenced code block."
)

_SQL_OPTIMIZE_SYSTEM = (
    "You are a database performance engineer. Review the given SQL for "
    "unnecessary joins, missing filters, SELECT *, non-sargable predicates, "
    "and indexing opportunities. List the concrete issues found, then "
    "provide an optimized rewrite in a single fenced SQL code block."
)

_TEST_GEN_SYSTEM = (
    "You are a Python testing expert. Write concise pytest unit tests for "
    "the given function or module, covering typical cases, edge cases, and "
    "at least one failure/exception case. Output only a single fenced "
    "Python code block containing the complete test file."
)


def _run(llm: BaseChatModel, system: str, content: str) -> str:
    if not content.strip():
        return "Please provide the code or query to work on."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=content)])
    return getattr(response, "content", str(response))


def make_refactor_tool(llm: BaseChatModel) -> Callable[[str], str]:
    def refactor_pandas_code(code: str) -> str:
        """Refactor slow Pandas code into a faster Polars or DuckDB equivalent."""
        return _run(llm, _REFACTOR_SYSTEM, code)

    return refactor_pandas_code


def make_sql_optimize_tool(llm: BaseChatModel) -> Callable[[str], str]:
    def optimize_sql(query: str) -> str:
        """Review a SQL query for performance issues and suggest an optimized rewrite."""
        return _run(llm, _SQL_OPTIMIZE_SYSTEM, query)

    return optimize_sql


def make_test_gen_tool(llm: BaseChatModel) -> Callable[[str], str]:
    def generate_pytest_tests(code: str) -> str:
        """Generate pytest unit tests for a given Python function or module."""
        return _run(llm, _TEST_GEN_SYSTEM, code)

    return generate_pytest_tests
