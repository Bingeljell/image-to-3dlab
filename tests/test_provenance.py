import pytest

from image_to_3dlab.provenance import LICENSES, validate_run_policy


def test_sf3d_is_allowed_when_conditionals_are_enabled():
    profile = validate_run_policy("sf3d", "game", "worldwide", True)
    assert profile.classification == "commercial-conditional"


def test_hunyuan_is_blocked_for_worldwide_game():
    with pytest.raises(ValueError, match="worldwide"):
        validate_run_policy("hunyuan-comfyui", "game", "worldwide", True)


def test_trellis_inherits_dinov3_conditional_classification():
    profile = validate_run_policy("trellis2", "game", "worldwide", True)
    assert profile.classification == "commercial-conditional"


def test_trellis_is_blocked_when_manifest_disallows_conditionals():
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy("trellis2", "game", "worldwide", False)


def test_qwen_image_is_conditional_not_research_only():
    """Qwen confirmed outputs are the user's and the model runs locally, so the only
    condition is the 'Built with Qwen' credit. An agent driving a game run with a
    manifest must not be refused for using it."""
    profile = LICENSES["qwen-image-2.1"]
    assert profile.classification == "commercial-conditional"
    assert profile.folder == "conditional"
    assert validate_run_policy("qwen-image-2.1", "game", "worldwide", allow_conditional=True) is profile
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy("qwen-image-2.1", "game", "worldwide", allow_conditional=False)
    assert any("Built with Qwen" in c for c in profile.conditions)


def test_qwen_outputs_are_yours_and_the_statement_is_cited():
    """Qwen confirmed on 2026-09-21 that outputs are not licensed Materials."""
    conditions = " ".join(LICENSES["qwen-image-2.1"].conditions)
    assert "https://x.com/QwenDevs/status/2101917379785838660" in conditions
    assert "are yours" in conditions
    assert "inherits the restriction" not in conditions


# No shipped model is research-only today (Qwen moved to conditional on 2026-10-07), but the
# gate stays for the next non-commercial model; these tests give it a stand-in profile.
@pytest.fixture
def research_model(monkeypatch):
    import dataclasses
    from image_to_3dlab import provenance
    stand_in = dataclasses.replace(LICENSES["qwen-image-2.1"], classification="research-only",
                                   folder="research_only")
    monkeypatch.setitem(provenance.LICENSES, "research-model", stand_in)
    return "research-model"


def test_research_only_is_refused_even_when_conditional_is_allowed(research_model):
    """allow_conditional is not consent to a non-commercial model. A manifest written for
    the Hunyuan or SF3D cases must not silently pick up a stricter licence."""
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(research_model, use_case="game", distribution="private",
                            allow_conditional=True)


def test_research_only_is_refused_for_public_distribution(research_model):
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(research_model, use_case="showcase", distribution="public",
                            allow_conditional=True)


def test_research_only_is_allowed_for_a_private_showcase(research_model):
    profile = validate_run_policy(research_model, use_case="showcase", distribution="private",
                                  allow_conditional=True)
    assert profile.classification == "research-only"


def test_research_only_still_needs_allow_conditional(research_model):
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy(research_model, use_case="showcase", distribution="private",
                            allow_conditional=False)


def test_qwen_wording_says_the_pictures_are_yours_without_warnings():
    """Qwen confirmed outputs are the user's (2026-09-21); the record says so plainly."""
    from image_to_3dlab.provenance import QWEN_OUTPUT_RIGHTS

    assert "are yours" in QWEN_OUTPUT_RIGHTS
    assert "ambiguous" not in QWEN_OUTPUT_RIGHTS and "non-commercial" not in QWEN_OUTPUT_RIGHTS.lower()
