from paper4_kbvqa.types import Evidence
from paper4_kbvqa.verification.llm_critic import (
    CriticLabel,
    build_critic_prompt,
    _normalise_log_scores,
)


def test_critic_prompt_has_no_gold_answer_channel():
    ev = Evidence("e1", "Paris is the capital of France.", "wikipedia", "https://example.org")
    prompt = build_critic_prompt(
        question="What is the capital of France?",
        answer="Paris",
        visual_entities=["France"],
        evidence=[ev],
        cited_evidence_ids=["e1"],
    )
    assert "GROUND_TRUTH" not in prompt
    assert "SUPPORTED" in prompt and "CONTRADICTED" in prompt and "INSUFFICIENT" in prompt
    assert "[e1]" in prompt


def test_label_probabilities_sum_to_one():
    p = _normalise_log_scores([-0.1, -1.2, -2.0])
    assert len(p) == 3
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[1] > p[2]


def test_label_enum_is_fixed():
    assert [x.value for x in CriticLabel] == [
        "SUPPORTED", "CONTRADICTED", "INSUFFICIENT"
    ]
