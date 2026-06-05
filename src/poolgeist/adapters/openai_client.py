"""Optional OpenAI adapter loaded from environment variables only."""

from __future__ import annotations

import os
from typing import Any


class OpenAIClientAdapter:
    """Thin optional OpenAI wrapper that never stores API keys in code."""

    def __init__(self, *, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL") or ""
        self.api_key_present = bool(os.getenv("OPENAI_API_KEY"))

    def is_configured(self) -> bool:
        """Return whether an API key and model are configured."""

        return self.api_key_present and bool(self.model)

    def client(self) -> Any:
        """Create an OpenAI client if the optional dependency is installed."""

        if not self.is_configured():
            raise RuntimeError("Set OPENAI_API_KEY and OPENAI_MODEL to enable OpenAI features.")
        from openai import OpenAI

        return OpenAI()
