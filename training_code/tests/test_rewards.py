from omni_opsd.rewards import extract_mcq_answer, mcq_exact_rewards


def test_extract_mcq_answer_accepts_common_short_forms():
    assert extract_mcq_answer("B") == "B"
    assert extract_mcq_answer("Option C.") == "C"
    assert extract_mcq_answer("<answer>D</answer>") == "D"
    assert extract_mcq_answer([{"role": "assistant", "content": "Answer: A"}]) == "A"


def test_extract_mcq_answer_does_not_match_arbitrary_words():
    assert extract_mcq_answer("Because the scene changes") is None
    assert extract_mcq_answer("") is None


def test_mcq_exact_rewards_are_binary_and_aligned():
    assert mcq_exact_rewards(["A", "Option C", "wrong"], ["A", "B", "D"]) == [
        1.0,
        0.0,
        0.0,
    ]
