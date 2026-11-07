import pytest
from pydantic import ValidationError

from app.schemas import AiResult, FgsChoices, ScoreSubmission
from app.scoring import calculate_fgs, combine_results


def choices(**overrides):
    return FgsChoices.model_validate({"ears": 0, "eyes": 0, "muzzle": 0, "whiskers": 0, "head": 0, **overrides})


@pytest.mark.parametrize(
    "label,raw,values,code",
    [
        ("pain", 0.8, {"ears": 2, "eyes": 2}, "agreement_positive"),
        ("no_pain", 0.2, {"ears": 2, "eyes": 2}, "fgs_positive"),
        ("pain", 0.8, {"ears": 1}, "discordant"),
        ("no_pain", 0.2, {}, "agreement_negative"),
    ],
)
def test_complete_comparison_matrix(label, raw, values, code):
    ai = AiResult(status="ok", label=label, raw_score=raw)
    assert combine_results(ai, calculate_fgs(choices(**values))).code == code


def test_na_keeps_partial_score_and_never_uses_complete_cutoff():
    fgs = calculate_fgs(choices(ears=2, eyes=2, head=None))
    assert fgs.total == 4
    assert fgs.max == 8
    assert fgs.assessable_count == 4
    assert fgs.normalized == 0.5
    assert not fgs.complete
    assert fgs.flag == "incomplete"
    result = combine_results(AiResult(status="unavailable"), fgs)
    assert result.code == "incomplete"
    assert "值得关注" in result.summary


def test_all_unassessable_has_no_division_by_zero():
    fgs = calculate_fgs(FgsChoices(ears=None, eyes=None, muzzle=None, whiskers=None, head=None))
    assert fgs.normalized is None
    assert fgs.total == fgs.max == fgs.assessable_count == 0


def test_unavailable_ai_preserves_positive_fgs():
    conclusion = combine_results(AiResult(status="unavailable"), calculate_fgs(choices(ears=2, eyes=2)))
    assert conclusion.code == "ai_unavailable"
    assert "达到关注参考值" in conclusion.title


@pytest.mark.parametrize("invalid", [True, False, 0.0, 1.5, "1", -1, 3, [], {}])
def test_scores_do_not_coerce_invalid_types(invalid):
    with pytest.raises(ValidationError):
        choices(eyes=invalid)


def test_missing_and_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        FgsChoices.model_validate({"ears": 0})
    with pytest.raises(ValidationError):
        ScoreSubmission.model_validate({"scores": choices().model_dump(), "ai": "no_pain"})


def test_invalid_model_outputs_are_not_reported_as_predictions():
    with pytest.raises(ValidationError):
        AiResult(status="ok", label="pain", raw_score=0.1)
    with pytest.raises(ValidationError):
        AiResult(status="ok", label="pain", raw_score=float("nan"))
    unavailable = AiResult(status="unavailable", label="pain", raw_score=0.9)
    assert unavailable.label is None
    assert unavailable.raw_score is None
