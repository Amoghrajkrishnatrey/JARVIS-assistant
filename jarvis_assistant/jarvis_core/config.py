"""
config.py
Centralized configuration for the JARVIS DS/BI Assistant.

All settings are read from environment variables (via a .env file, see
.env.example) so the assistant can be reconfigured without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM backend ---------------------------------------------------
    llm_backend: str = os.getenv("JARVIS_LLM_BACKEND", "ollama")  # "ollama" | "cloud"
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    cloud_provider: str = os.getenv("JARVIS_CLOUD_PROVIDER", "anthropic")  # "anthropic" | "openai"
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --- Voice -----------------------------------------------------------
    voice_enabled: bool = _bool_env("JARVIS_VOICE_ENABLED", "true")
    speech_rate: int = int(os.getenv("JARVIS_SPEECH_RATE", "165"))
    speech_max_chars: int = int(os.getenv("JARVIS_SPEECH_MAX_CHARS", "400"))

    # --- Paths -------------------------------------------------------------
    workspace_dir: Path = Path(os.getenv("JARVIS_WORKSPACE", "./jarvis_workspace")).resolve()
    vector_store_dir: Path = Path(
        os.getenv("JARVIS_VECTOR_STORE", "./jarvis_workspace/chroma_db")
    ).resolve()
    data_dir: Path = Path(os.getenv("JARVIS_DATA_DIR", "./jarvis_workspace/data")).resolve()

    # --- Execution safety ----------------------------------------------
    exec_timeout_seconds: int = int(os.getenv("JARVIS_EXEC_TIMEOUT", "20"))

    # --- Legacy / optional APIs -----------------------------------------
    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")

    def ensure_dirs(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
