"""Optional OpenAI chat-completions adapter for structured news extraction."""

from __future__ import annotations

import importlib.util
import os
from typing import Any


class OpenAIResponsesAdapter:
    """Thin optional adapter that disables itself unless key, model, and package exist."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL")
        self.enabled = bool(self.api_key and self.model)
        self.warning: str | None = None
        self.client: Any | None = None

        if not self.enabled:
            self.warning = (
                "OPENAI_API_KEY and OPENAI_MODEL are required; LLM news extraction is disabled."
            )
            return
        if importlib.util.find_spec("openai") is None:
            self.warning = "OpenAI package is not installed; LLM news extraction is disabled."
            self.enabled = False
            return

        from openai import OpenAI  # noqa: PLC0415

        self.client = OpenAI(api_key=self.api_key)

    def extract_structured_signal(self, text: str) -> dict[str, Any] | None:
        """Extract structured news signals via chat completions, returning None when disabled."""

        if not self.enabled or self.client is None or self.model is None:
            return None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only structured football team news modifiers. "
                        "Do not make predictions."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        content = response.choices[0].message.content
        return {"raw_text": content or "", "model": self.model}
