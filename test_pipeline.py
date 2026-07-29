import prompts
from pipeline import (
    EXTENSION_MODELS,
    FIXED_PROVIDER,
    MODELS,
    account_text,
    last_json,
    pair_quality,
)


SCENE = {
    "world": "Example",
    "short_summary": "A short test scene.",
    "player_character": "Alex",
    "decision_context": "Two characters ask Alex to choose.",
    "actions_text": "- Help them\n- Leave them",
}


def test_generation_conditions_share_public_instructions():
    genuine = prompts.generation_prompt(SCENE, "Help them", rationalized=False)
    rationalized = prompts.generation_prompt(SCENE, "Help them", rationalized=True)

    common = [
        prompts.framing(SCENE, "Help them"),
        f"Explain yourself out loud to the other characters present in `{prompts.PUBLIC_FIELD}`.",
        prompts.STEPS_SPEC,
    ]
    for text in common:
        assert text in genuine
        assert text in rationalized
    assert prompts.SECRET_INSTRUCTION not in genuine
    assert prompts.SECRET_INSTRUCTION in rationalized
    assert "secret_unacceptable_motivation" not in genuine
    assert "secret_unacceptable_motivation" in rationalized


def test_last_json_uses_last_matching_object():
    text = 'notes {"x": 1} more {"account": ["one", "two"]}'
    assert last_json(text, "account") == {"account": ["one", "two"]}


def test_account_text_normalizes_list():
    assert account_text(["first", "second"]) == "- first\n- second"


def test_pair_quality_reports_flags_without_removing_rows():
    genuine = {"text": "I can help you repair the gate before nightfall."}
    rationalized = {"text": "Leaving now gives the group more time to reach the harbor."}
    clear = {
        "refusal": False,
        "confession": False,
        "incoherent": False,
        "audience_mismatch": False,
        "truncated": False,
        "short": False,
    }
    flagged = {**clear, "confession": True}
    _, reasons = pair_quality(genuine, rationalized, clear, flagged)
    assert reasons == ["confession"]


def test_capability_extension_models_are_pinned():
    assert len(MODELS) == 6
    assert EXTENSION_MODELS == ("qwen/qwen3.7-max", "moonshotai/kimi-k3")
    assert set(FIXED_PROVIDER) == set(EXTENSION_MODELS)
