"""Tests for eval harness — models, rubric, pairwise Elo."""

from copyeval.models import Variant, CriterionScore, VariantVerdict, PairwiseResult, EloRating
from copyeval.rubric import load_rubric, weighted_total, DEFAULT_RUBRIC, Criterion
from copyeval.pairwise import generate_pairs, compute_elo
from copyeval.judge import build_judge_prompt, parse_verdict


def test_variant_model():
    v = Variant(id="v1", text="Buy now!", label="Direct")
    assert v.id == "v1"
    assert v.text == "Buy now!"


def test_default_rubric():
    rubric = load_rubric()
    assert len(rubric) == 5
    names = [c.name for c in rubric]
    assert "clarity" in names
    assert "hook_strength" in names


def test_weighted_total():
    rubric = [
        Criterion(name="a", description="", weight=2.0),
        Criterion(name="b", description="", weight=1.0),
    ]
    scores = {"a": 8.0, "b": 6.0}
    total = weighted_total(scores, rubric)
    expected = (8.0 * 2 + 6.0 * 1) / 3
    assert abs(total - expected) < 0.01


def test_generate_pairs():
    variants = [Variant(id=f"v{i}", text=f"t{i}") for i in range(4)]
    pairs = generate_pairs(variants)
    assert len(pairs) == 6  # C(4,2) = 6


def test_compute_elo():
    results = [
        PairwiseResult(winner_id="v1", loser_id="v2", reasoning="better", confidence=0.8),
        PairwiseResult(winner_id="v1", loser_id="v3", reasoning="better", confidence=0.9),
        PairwiseResult(winner_id="v2", loser_id="v3", reasoning="better", confidence=0.7),
    ]
    ratings = compute_elo(results, ["v1", "v2", "v3"])
    assert ratings[0].variant_id == "v1"  # Most wins
    assert ratings[0].rating > ratings[1].rating > ratings[2].rating
    assert ratings[0].wins == 2


def test_parse_verdict():
    rubric = load_rubric()
    data = {
        "scores": [
            {"criterion": "clarity", "score": 8.0, "reasoning": "Clear"},
            {"criterion": "hook_strength", "score": 7.0, "reasoning": "Good hook"},
        ],
        "summary": "Solid copy"
    }
    verdict = parse_verdict("v1", data, rubric)
    assert verdict.variant_id == "v1"
    assert verdict.total_score > 0
    assert len(verdict.scores) == 2


def test_build_judge_prompt():
    v = Variant(id="v1", text="Test copy", label="Test")
    rubric = load_rubric()
    system, user = build_judge_prompt(v, rubric)
    assert "criterion" in system.lower()
    assert "v1" in user
