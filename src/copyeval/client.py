"""Anthropic API client for eval calls."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()


def get_client():
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("sk-ant-your"):
        raise ValueError("ANTHROPIC_API_KEY not set. Copy .env.example to .env.")
    return anthropic.Anthropic(api_key=key)


def call_judge(system: str, user: str, temperature: float = 0.0) -> dict:
    """Call the LLM and parse JSON response."""
    client = get_client()
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "\n".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
    if not text:
        raise ValueError("Judge returned no text")
    # Extract JSON from potential markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("Judge must return a JSON object")
    return data
