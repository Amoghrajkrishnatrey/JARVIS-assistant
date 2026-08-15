"""
llm_backend.py
Dual LLM backend resolution: local Ollama by default, with an optional
cloud fallback (Anthropic or OpenAI) when Ollama is unreachable or a cloud
backend is explicitly requested.

Returns a LangChain-compatible chat model so it can be bound to the same
LangGraph agent regardless of which backend actually served the request.
"""
from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from .config import settings

logger = logging.getLogger("jarvis.llm")


class LLMUnavailableError(RuntimeError):
    """Raised when neither the local nor the cloud LLM backend is reachable."""


def _build_ollama() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
    llm.invoke("ping")  # cheap connectivity probe; raises if Ollama is down
    return llm


def _build_cloud() -> BaseChatModel:
    if settings.cloud_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise LLMUnavailableError("OPENAI_API_KEY not set for cloud fallback.")
        return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.2)

    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise LLMUnavailableError("ANTHROPIC_API_KEY not set for cloud fallback.")
    return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key, temperature=0.2)


def get_llm() -> BaseChatModel:
    """
    Resolve an LLM instance according to JARVIS_LLM_BACKEND.

    - "cloud": go straight to the configured cloud provider.
    - "ollama" (default): try local Ollama first; if it's unreachable,
      fall back to the cloud provider if credentials are configured.
    """
    preferred = settings.llm_backend.lower()

    if preferred == "cloud":
        logger.info("Using cloud LLM backend ('%s') by configuration.", settings.cloud_provider)
        return _build_cloud()

    try:
        logger.info("Connecting to local Ollama model '%s' at %s ...", settings.ollama_model, settings.ollama_base_url)
        return _build_ollama()
    except Exception as exc:
        logger.warning("Local Ollama backend unavailable (%s). Trying cloud fallback...", exc)
        try:
            return _build_cloud()
        except Exception as cloud_exc:
            raise LLMUnavailableError(
                "No LLM backend available.\n"
                f"  - Local: start Ollama ('ollama serve') and pull a model "
                f"('ollama pull {settings.ollama_model}'), or\n"
                "  - Cloud: set ANTHROPIC_API_KEY or OPENAI_API_KEY and "
                "JARVIS_LLM_BACKEND=cloud in your .env file."
            ) from cloud_exc
