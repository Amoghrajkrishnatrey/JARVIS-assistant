"""
jarvis.py
JARVIS DS/BI Assistant — CLI entry point.

    python jarvis.py

See README.md for setup instructions (Ollama, optional cloud fallback,
and required pip packages).
"""
from __future__ import annotations

import logging
import uuid

from jarvis_core.agent import build_agent
from jarvis_core.config import settings
from jarvis_core.llm_backend import LLMUnavailableError
from jarvis_core.voice import Voice

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("jarvis")

BANNER = r"""
============================================================
  JARVIS  |  Data Science & Business Intelligence Assistant
============================================================
"""

HELP_TEXT = """
Examples of what you can ask:
  - "Scaffold a new project called churn_model"
  - "Load sales.csv and give me an EDA summary"
  - "Run: SELECT category, SUM(revenue) FROM sales GROUP BY category"
  - "Register the file ~/data/orders.parquet"
  - "Refactor this pandas code to Polars: <code>"
  - "Optimize this SQL: <query>"
  - "Generate pytest tests for this function: <code>"
  - "Remember: DAU means Daily Active Users, counted as..."
  - "What does DAU mean?"                (searches the data dictionary)
  - "What's the time / date / weather in Mumbai / tell me a joke"

Type 'exit' or 'quit' to leave.
"""


class JarvisCLI:
    """Owns the voice front end and the agent session for one run."""

    def __init__(self) -> None:
        self.voice = Voice()
        try:
            self.agent = build_agent()
        except LLMUnavailableError as exc:
            logger.error(str(exc))
            raise SystemExit(1)
        # A stable thread_id gives the LangGraph checkpointer continuity
        # across turns within this single CLI session.
        self.thread_id = str(uuid.uuid4())

    def run(self) -> None:
        print(BANNER)
        self.voice.say("Good day! JARVIS is online and ready to work with your data.")
        print(HELP_TEXT)
        while True:
            try:
                user_input = input("👤 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                self.voice.say("Shutting down. Goodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "goodbye", "bye"}:
                self.voice.say("Goodbye. It was a pleasure serving you.")
                break
            if user_input.lower() in {"help", "commands"}:
                print(HELP_TEXT)
                continue

            self._handle(user_input)

    def _handle(self, user_input: str) -> None:
        config = {"configurable": {"thread_id": self.thread_id}}
        try:
            result = self.agent.invoke({"messages": [("user", user_input)]}, config=config)
            final_message = result["messages"][-1].content
        except Exception as exc:
            final_message = f"I hit an error handling that request: {exc}"
            logger.exception("Agent execution failed")
        self.voice.say(final_message)


if __name__ == "__main__":
    settings.ensure_dirs()
    JarvisCLI().run()
