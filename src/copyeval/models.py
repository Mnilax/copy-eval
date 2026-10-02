"""Pydantic models for evaluation data."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Variant(BaseModel):
    """A copy variant to evaluate."""
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: str = ""


class CriterionScore(BaseModel):
    """Score for a single criterion."""
    criterion: str
    score: float = Field(ge=0, le=10)
    reasoning: str


class VariantVerdict(BaseModel):
    """Full verdict for one variant."""
    variant_id: str
    scores: list[CriterionScore]
    total_score: float
    summary: str


class PairwiseResult(BaseModel):
    """Result of a pairwise comparison."""
    winner_id: str
    loser_id: str
    reasoning: str
    confidence: float = Field(ge=0, le=1)


class EloRating(BaseModel):
    """Elo rating for a variant."""
    variant_id: str
    rating: float
    wins: int = 0
    losses: int = 0


class EvalReport(BaseModel):
    """Complete evaluation report."""
    verdicts: list[VariantVerdict] = []
    elo_ratings: list[EloRating] = []
    pairwise_results: list[PairwiseResult] = []
