"""
voice.py
Thin wrapper around pyttsx3 with graceful degradation. Truncates and
strips code blocks/tables before speaking so JARVIS doesn't try to read
an entire dataframe or stack trace aloud; the full text is still printed.
"""
from __future__ import annotations

import logging
import re

from .config import settings

logger = logging.getLogger("jarvis.voice")


class Voice:
    """Text-to-speech front end. Falls back to text-only mode on any
    initialization failure (e.g. no audio device available)."""

    def __init__(self) -> None:
        self.enabled = settings.voice_enabled
        self._engine = None
        if self.enabled:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", settings.speech_rate)
                self._engine.setProperty("volume", 1.0)
            except Exception as exc:
                logger.warning("TTS engine unavailable, continuing in text-only mode: %s", exc)
                self.enabled = False

    @staticmethod
    def _speakable(text: str, max_chars: int) -> str:
        """Produce a short, speech-friendly version of a (possibly long,
        code-containing) response."""
        clean = re.sub(r"```.*?```", " code omitted, see printed output. ", text, flags=re.S)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) > max_chars:
            clean = clean[:max_chars].rsplit(" ", 1)[0] + "... see full output above."
        return clean

    def say(self, text: str) -> None:
        """Print the full response and speak a trimmed version of it."""
        print(f"\n🤖 JARVIS: {text}\n")
        if not self.enabled or not self._engine:
            return
        try:
            spoken = self._speakable(text, settings.speech_max_chars)
            self._engine.say(spoken)
            self._engine.runAndWait()
        except Exception as exc:
            logger.warning("Speech playback failed: %s", exc)
