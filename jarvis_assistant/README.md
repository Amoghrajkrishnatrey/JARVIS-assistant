# JARVIS — Data Science & BI Assistant

A local, voice-enabled AI assistant specialized in Data Science, Advanced
Analytics, and Business Intelligence. Built on a LangGraph tool-calling
agent instead of keyword matching, with a dual local/cloud LLM backend.

## Architecture

```
jarvis.py                      # CLI entry point
jarvis_core/
  config.py                    # env-driven settings
  llm_backend.py                # Ollama (local) with cloud fallback
  voice.py                     # pyttsx3 wrapper
  agent.py                     # LangGraph agent + tool registry
  tools/
    code_execution.py          # sandboxed Python/pandas/polars execution
    sql_engine.py               # DuckDB query engine
    eda.py                      # exploratory data analysis
    scaffolding.py               # Cookiecutter-DS project generator
    code_enhancement.py          # LLM-driven refactor / SQL optimize / test gen
    rag_memory.py                 # ChromaDB data dictionary (RAG)
    utilities.py                 # time, date, weather, jokes, safe calculator
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pyttsx3` needs a system TTS engine: `espeak` on most Linux distros
(`sudo apt install espeak`), and works out of the box on macOS/Windows.
If no engine is found, JARVIS automatically falls back to text-only mode.

### 2. Set up the local LLM (Ollama)

```bash
# Install Ollama: https://ollama.com/download
ollama serve
ollama pull qwen2.5-coder:latest     # or llama3, etc.
```

### 3. (Optional) configure a cloud fallback

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`. JARVIS uses Ollama by default and only calls the cloud
API if Ollama is unreachable, or if you set `JARVIS_LLM_BACKEND=cloud`.

### 4. Run

```bash
python jarvis.py
```

Datasets you want JARVIS to auto-discover for SQL/EDA can be dropped into
`jarvis_workspace/data/` (CSV or Parquet); it also accepts an absolute
path to any file on demand.

## What changed from the original script

- **Agentic orchestration** replaces the `if/elif` command parser: a
  LangGraph `create_react_agent` decides which tool(s) to call from
  natural language, with short-term conversational memory via
  `MemorySaver`.
- **Dual LLM backend**: local Ollama by default, with automatic (or
  forced) fallback to Anthropic/OpenAI.
- **No more raw `eval()`**: arithmetic goes through an AST-based safe
  evaluator; general data-analysis code runs in an isolated subprocess
  with a restricted builtin/import surface, a wall-clock timeout, and
  (on POSIX) CPU/memory limits. This is defense-in-depth for a local,
  single-user tool — swap in Docker/gVisor if you ever expose it to
  untrusted input or multiple users.
- **DuckDB engine** for ad hoc SQL against local CSV/Parquet files.
- **EDA tool** for shape/dtypes/missing-values/summary-stats profiling.
- **Project scaffolding** tool for a Cookiecutter-DS-style layout.
- **Code enhancement tools**: pandas→Polars/DuckDB refactors, SQL
  optimization review, and pytest generation — all LLM-driven since
  good refactors are context-dependent, not rule-based.
- **Local RAG data dictionary** (ChromaDB + sentence-transformers) so
  JARVIS can learn and recall your team's KPI/schema definitions.
- Legacy utilities (time, date, weather, jokes) were kept, ported into
  the new tool interface.

## Notes & limitations

- The code-execution sandbox blocks imports outside a small data-science
  whitelist (pandas, numpy, polars, sklearn, matplotlib, seaborn, plus a
  few stdlib modules) and disables `open`, `eval`, `exec`, and file/network
  access from within the executed snippet — but it is still a same-machine
  subprocess, not a hardened container. Don't point it at untrusted code.
- `create_react_agent`'s tool-calling quality depends on the underlying
  model's function-calling support. Smaller local models may need more
  explicit prompts than a top-tier cloud model would.
- ChromaDB's `PersistentClient` and `sentence-transformers` download a
  small embedding model (`all-MiniLM-L6-v2`) on first use — this requires
  network access once, then works offline.
