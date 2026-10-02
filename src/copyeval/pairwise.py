"""Pairwise comparisons + Elo rating system."""

from __future__ import annotations

import math
from itertools import combinations

from copyeval.models import EloRating, PairwiseResult, Variant

K_FACTOR = 32
INITIAL_ELO = 1500.0


def build_pairwise_prompt(a: Variant, b: Variant) -> tuple[str, str]:
    """Build prompt for pairwise comparison."""
    system = """You are an expert marketing copy evaluator. Compare two copy variants and pick the better one.

Respond with ONLY valid JSON (no markdown, no preamble):
{
  "winner": "A" or "B",
  "reasoning": "one sentence why the winner is better",
  "confidence": 0.0-1.0
}"""

    user = f"""Compare these two copy variants:

--- Variant A (id: {a.id}) ---
{a.text}

--- Variant B (id: {b.id}) ---
{b.text}

Which is better overall for marketing effectiveness?"""

    return system, user


def parse_pairwise(a_id: str, b_id: str, data: dict) -> PairwiseResult:
    """Parse pairwise result JSON."""
    winner = data["winner"]
    if not isinstance(winner, str) or winner.strip().upper() not in ("A", "B"):
        raise ValueError("Judge winner must be A or B")
    if a_id == b_id:
        raise ValueError("Pairwise variants must have different IDs")
    winner = winner.strip().upper()
    winner_id = a_id if winner == "A" else b_id
    loser_id = b_id if winner == "A" else a_id

    return PairwiseResult(
        winner_id=winner_id,
        loser_id=loser_id,
        reasoning=data.get("reasoning", ""),
        confidence=data.get("confidence", 0.5),
    )


def generate_pairs(variants: list[Variant]) -> list[tuple[Variant, Variant]]:
    """Generate all unique pairs for comparison."""
    return list(combinations(variants, 2))


def compute_elo(results: list[PairwiseResult], variant_ids: list[str]) -> list[EloRating]:
    """Compute Elo ratings from pairwise results."""
    ratings = {vid: INITIAL_ELO for vid in variant_ids}
    wins = {vid: 0 for vid in variant_ids}
    losses = {vid: 0 for vid in variant_ids}

    for result in results:
        w, l = result.winner_id, result.loser_id
        if w not in ratings or l not in ratings:
            continue

        # Expected scores
        exp_w = 1 / (1 + math.pow(10, (ratings[l] - ratings[w]) / 400))
        exp_l = 1 - exp_w

        # Update
        ratings[w] += K_FACTOR * (1 - exp_w)
        ratings[l] += K_FACTOR * (0 - exp_l)
        wins[w] += 1
        losses[l] += 1

    return sorted(
        [EloRating(variant_id=vid, rating=round(ratings[vid], 1), wins=wins[vid], losses=losses[vid])
         for vid in variant_ids],
        key=lambda x: x.rating,
        reverse=True,
    )
