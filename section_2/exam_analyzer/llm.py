"""LLM client wrapper. Model-swappable; temperature=0 for reproducible runs."""
from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_MODEL = os.environ.get("ANALYZER_MODEL", "claude-sonnet-4-6")


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = None  # lazy init so importing the module needs no key

    def _ensure(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> Any:
        """Call the model and parse a single JSON object/array from the reply."""
        client = self._ensure()
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _parse_json(text)


def _parse_json(text: str) -> Any:
    """Robustly extract the first JSON value from a model response."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fall back: grab the outermost {...} or [...]
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in model output:\n{text[:500]}")
        return json.loads(match.group(1))
