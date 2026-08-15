"""
agent.py
Agentic orchestration layer. Replaces the legacy if/elif string-matching
command parser with a LangGraph tool-calling ReAct agent: the LLM decides
which tool(s) to invoke based on the user's natural-language request,
rather than JARVIS pattern-matching keywords.
"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .llm_backend import get_llm
from .tools import code_enhancement, eda, scaffolding, utilities
from .tools.code_execution import run_python_snippet
from .tools.rag_memory import data_dictionary
from .tools.sql_engine import engine as sql_engine

SYSTEM_PROMPT = """You are JARVIS, a local AI assistant specialized in Data \
Science, Advanced Analytics, and Business Intelligence.

You have tools for: sandboxed Python/pandas/polars/scikit-learn execution, \
DuckDB SQL queries over local CSV/Parquet files, exploratory data analysis, \
project scaffolding, Pandas-to-Polars/DuckDB refactoring, SQL optimization \
review, pytest test generation, a local data-dictionary memory (RAG), and \
small utilities (time, date, weather, jokes, arithmetic).

Rules:
- Prefer calling a tool over guessing whenever the request involves real \
data, files, code execution, SQL, or calculations.
- When a tool returns an error, explain it plainly and suggest a concrete \
fix; never claim it succeeded if it didn't.
- Keep the final answer concise and conversational (it may be read aloud); \
tool output with tables or code is already shown to the user, so summarize \
it rather than repeating it verbatim.
"""


# ---------------------------------------------------------------------------
# Tool wrappers — thin @tool-decorated functions around the modules in
# jarvis_core/tools/, so the agent sees a clean, documented tool surface.
# ---------------------------------------------------------------------------

@tool
def python_data_tool(code: str) -> str:
    """Execute Python code for data analysis (pandas, numpy, polars,
    scikit-learn, matplotlib, seaborn are available) in an isolated
    subprocess sandbox. The code must print() anything it wants returned.
    Returns captured stdout, or a formatted error on failure."""
    result = run_python_snippet(code)
    if result.success:
        return result.stdout or "(code ran successfully with no printed output)"
    return f"Execution failed:\n{result.stderr}"


@tool
def sql_query_tool(query: str) -> str:
    """Run a SQL query with DuckDB against locally registered CSV/Parquet
    files. Returns a preview table (first 15 rows)."""
    return sql_engine.query(query).preview()


@tool
def list_datasets_tool() -> str:
    """List tables/views currently queryable via the DuckDB engine."""
    tables = sql_engine.list_tables()
    return "\n".join(tables) if tables else "No datasets registered yet."


@tool
def register_dataset_tool(path: str) -> str:
    """Register a CSV or Parquet file at the given path as a queryable
    DuckDB view, so it can be used in sql_query_tool."""
    return sql_engine.register_file(path)


@tool
def eda_tool(filename: str) -> str:
    """Load a CSV/Parquet file and return its shape, column dtypes,
    missing-value counts, duplicate-row count, and summary statistics."""
    return eda.load_and_profile(filename)


@tool
def scaffold_project_tool(project_name: str) -> str:
    """Generate a standardized Cookiecutter-Data-Science-style project
    folder structure (data/raw, data/processed, notebooks, src, models,
    reports) with a tailored requirements.txt and .gitignore."""
    return scaffolding.scaffold_project(project_name)


@tool
def data_dictionary_search_tool(query: str) -> str:
    """Semantic search over indexed schema/KPI/metric definitions in the
    local data dictionary."""
    return data_dictionary.search(query)


@tool
def data_dictionary_add_tool(term: str, definition: str) -> str:
    """Add a single term/definition pair to the local data dictionary
    (e.g. term='DAU', definition='Daily Active Users, counted as...')."""
    return data_dictionary.add_entry(term, definition)


@tool
def get_time_tool() -> str:
    """Get the current local time."""
    return utilities.get_time()


@tool
def get_date_tool() -> str:
    """Get today's date."""
    return utilities.get_date()


@tool
def get_weather_tool(city: str = "London") -> str:
    """Get current weather conditions for a city."""
    return utilities.get_weather(city)


@tool
def tell_joke_tool() -> str:
    """Tell a random joke."""
    return utilities.tell_joke()


@tool
def calculate_tool(expression: str) -> str:
    """Safely evaluate a plain arithmetic expression, e.g. '(12 + 3) * 2'."""
    return utilities.calculate(expression)


def build_agent():
    """Resolve the configured LLM and construct the LangGraph tool-calling
    agent with the full DS/BI tool suite bound to it."""
    llm = get_llm()

    tools = [
        python_data_tool, sql_query_tool, list_datasets_tool, register_dataset_tool,
        eda_tool, scaffold_project_tool,
        data_dictionary_search_tool, data_dictionary_add_tool,
        get_time_tool, get_date_tool, get_weather_tool, tell_joke_tool, calculate_tool,
        # LLM-driven tools need the resolved model injected via closures.
        tool(code_enhancement.make_refactor_tool(llm)),
        tool(code_enhancement.make_sql_optimize_tool(llm)),
        tool(code_enhancement.make_test_gen_tool(llm)),
    ]

    checkpointer = MemorySaver()  # gives the agent short-term conversational memory
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
