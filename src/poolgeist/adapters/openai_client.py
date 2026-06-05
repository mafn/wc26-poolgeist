"""Optional OpenAI Responses API adapter for structured news extraction."""

from __future__ import annotations

import os
from typing import Any


class OpenAIResponsesAdapter:
    """Thin optional adapter that disables itself unless key, model, and package exist."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL")
        self.enabled = bool(self.api_key and self.model)
        self.warning: str | None = None
        try:
            from openai import OpenAI  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            self.client = None
            if self.enabled:
                self.warning = "OpenAI package is not installed; LLM news extraction is disabled."
            self.enabled = False
        else:
            self.client = OpenAI(api_key=self.api_key) if self.enabled else None
            if not self.enabled:
                self.warning = (
                    "OPENAI_API_KEY and OPENAI_MODEL are required; LLM news extraction is disabled."
                )

    def extract_structured_signal(self, text: str) -> dict[str, Any] | None:
        """Extract structured news signals, returning None when disabled."""

        if not self.enabled or self.client is None or self.model is None:
            return None
        response = self.client.responses.create(
            model=self.model,
            input=(
                "Extract only structured football team news modifiers. Do not make predictions.\n"
                + text
            ),
        )
        return {"raw_text": getattr(response, "output_text", ""), "model": self.model}
