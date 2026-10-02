"""Evaluation rubric — criteria and weights."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Criterion:
    """A single evaluation criterion."""
    name: str
    description: str
    weight: float = 1.0


DEFAULT_RUBRIC: list[Criterion] = [
    Criterion(
        name="clarity",
        description="How clear and easy to understand is the copy? No jargon confusion, logical flow.",
        weight=1.0,
    ),
    Criterion(
        name="hook_strength",
        description="How compelling is the opening? Does it stop the scroll and create curiosity?",
        weight=1.5,
    ),
    Criterion(
        name="cta_effectiveness",
        description="How clear and compelling is the call to action? Does the reader know what to do next?",
        weight=1.2,
    ),
    Criterion(
        name="brand_fit",
        description="Does the tone, vocabulary, and positioning match the brand voice and target audience?",
        weight=1.0,
    ),
    Criterion(
        name="emotional_resonance",
        description="Does the copy evoke the intended emotion? Does it connect with pain points or aspirations?",
        weight=0.8,
    ),
]


def load_rubric(criteria: list[dict] | None = None) -> list[Criterion]:
    """Load rubric from dict list or return defaults."""
    if not criteria:
        return DEFAULT_RUBRIC
    return [Criterion(**c) for c in criteria]


def weighted_total(scores: dict[str, float], rubric: list[Criterion]) -> float:
    """Calculate weighted total score."""
    total = 0.0
    weight_sum = 0.0
    for c in rubric:
        if c.name in scores:
            total += scores[c.name] * c.weight
            weight_sum += c.weight
    return total / weight_sum if weight_sum > 0 else 0.0
