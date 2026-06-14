"""LLM-as-judge — rubric scoring with structured JSON output."""

from __future__ import annotations
from copyeval.models import Variant, VariantVerdict, CriterionScore
from copyeval.rubric import Criterion, weighted_total


def build_judge_prompt(variant: Variant, rubric: list[Criterion]) -> tuple[str, str]:
    """Build system + user prompts for rubric evaluation."""
    criteria_desc = "\n".join(
        f"- {c.name}: {c.description} (weight: {c.weight})"
        for c in rubric
    )

    system = f"""You are an expert marketing copy evaluator. Score the given copy variant on each criterion (0-10).

Criteria:
{criteria_desc}

Respond with ONLY valid JSON (no markdown, no preamble):
{{
  "scores": [
    {{"criterion": "name", "score": 7.5, "reasoning": "one sentence why"}},
    ...
  ],
  "summary": "one-sentence overall verdict"
}}"""

    user = f"""Evaluate this copy variant:

ID: {variant.id}
Label: {variant.label}
Text:
{variant.text}"""

    return system, user


def parse_verdict(variant_id: str, data: dict, rubric: list[Criterion]) -> VariantVerdict:
    """Parse LLM JSON response into a VariantVerdict."""
    scores = [CriterionScore(**s) for s in data["scores"]]
    score_map = {s.criterion: s.score for s in scores}
    total = weighted_total(score_map, rubric)

    return VariantVerdict(
        variant_id=variant_id,
        scores=scores,
        total_score=round(total, 2),
        summary=data.get("summary", ""),
    )
