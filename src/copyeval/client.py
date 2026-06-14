"""Anthropic API client for eval calls."""

from __future__ import annotations
import os, json
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
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text.strip()
    # Extract JSON from potential markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)
