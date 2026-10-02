import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from copyeval.cli import app, _load_variants
from copyeval.judge import parse_verdict
from copyeval.pairwise import parse_pairwise
from copyeval.rubric import load_rubric
from copyeval.client import call_judge


def judge_response(value):
    return {"scores": [{"criterion": c.name, "score": value, "reasoning": "mock"} for c in load_rubric()], "summary": "mock"}


def test_score_rounds_call_each_time_and_average(tmp_path, monkeypatch):
    variants = tmp_path / "variants.json"
    variants.write_text(json.dumps({"variants": [{"id": "v1", "text": "copy"}]}), encoding="utf-8")
    calls = []
    def call_judge(system, user):
        calls.append(user)
        return judge_response(2 if len(calls) == 1 else 8)
    monkeypatch.setattr("copyeval.client.call_judge", call_judge)
    result = CliRunner().invoke(app, ["score", str(variants), "--rounds", "2"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert "5.0" in result.output


def test_rounds_must_be_positive():
    result = CliRunner().invoke(app, ["score", "missing.yaml", "--rounds", "0"])
    assert result.exit_code == 2


@pytest.mark.parametrize("winner", ["tie", "C", "", None, 2])
def test_invalid_pairwise_winner_never_becomes_b(winner):
    with pytest.raises(ValueError, match="winner"):
        parse_pairwise("a", "b", {"winner": winner})


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_incomplete_or_ambiguous_rubric_is_rejected(change):
    data = judge_response(8)
    if change == "missing":
        data["scores"].pop()
    elif change == "duplicate":
        data["scores"].append(data["scores"][0])
    else:
        data["scores"][0]["criterion"] = "invented"
    with pytest.raises(ValueError, match="every rubric criterion"):
        parse_verdict("v1", data, load_rubric())


def test_duplicate_variant_ids_rejected(tmp_path):
    path = tmp_path / "variants.json"
    path.write_text(json.dumps({"variants": [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}]}))
    with pytest.raises(Exception, match="unique"):
        _load_variants(path)


def test_utf8_variants(tmp_path):
    path = tmp_path / "variants.yaml"
    path.write_text('variants:\n  - id: a\n    text: "Привет мир"\n', encoding="utf-8")
    assert _load_variants(path)[0].text == "Привет мир"


def test_judge_uses_text_block_after_nontext_content(monkeypatch):
    response = SimpleNamespace(content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text='```json\n{"winner":"A"}\n```')])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr("copyeval.client.get_client", lambda: client)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert call_judge("system", "user") == {"winner": "A"}
    assert calls[-1]["model"] == "claude-sonnet-4-6"
    monkeypatch.setenv("ANTHROPIC_MODEL", "configured-model")
    call_judge("system", "user")
    assert calls[-1]["model"] == "configured-model"


def test_judge_rejects_non_object_json(monkeypatch):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="[]")])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    monkeypatch.setattr("copyeval.client.get_client", lambda: client)
    with pytest.raises(TypeError, match="JSON object"):
        call_judge("system", "user")
