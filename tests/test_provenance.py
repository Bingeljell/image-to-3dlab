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


def test_qwen_image_is_classified_research_only():
    """Qwen-Image-2.1 is under the Qwen Research License: non-commercial only. It is the
    first non-commercial model in this repo and needed a class of its own, because
    `commercial-conditional` and `territory-restricted` both permit commercial use under
    conditions and this permits none."""
    profile = LICENSES["qwen-image-2.1"]
    assert profile.classification == "research-only"
    assert profile.folder == "research_only"
    assert any("Built with Qwen" in c for c in profile.conditions)


def test_qwen_outputs_are_yours_and_the_statement_is_cited():
    """Qwen confirmed on 2026-09-21 that outputs are not licensed Materials."""
    conditions = " ".join(LICENSES["qwen-image-2.1"].conditions)
    assert "https://x.com/QwenDevs/status/2101917379785838660" in conditions
    assert "are yours" in conditions
    assert "inherits the restriction" not in conditions


def test_research_only_is_refused_even_when_conditional_is_allowed():
    """allow_conditional is not consent to a non-commercial model. A manifest written for
    the Hunyuan or SF3D cases must not silently pick up a stricter licence."""
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(
            "qwen-image-2.1", use_case="game", distribution="private",
            allow_conditional=True,
        )


def test_research_only_is_refused_for_public_distribution():
    with pytest.raises(ValueError, match="research-only"):
        validate_run_policy(
            "qwen-image-2.1", use_case="showcase", distribution="public",
            allow_conditional=True,
        )


def test_research_only_is_allowed_for_a_private_showcase():
    profile = validate_run_policy(
        "qwen-image-2.1", use_case="showcase", distribution="private",
        allow_conditional=True,
    )
    assert profile.classification == "research-only"


def test_research_only_still_needs_allow_conditional():
    with pytest.raises(ValueError, match="disallows conditional"):
        validate_run_policy(
            "qwen-image-2.1", use_case="showcase", distribution="private",
            allow_conditional=False,
        )


def test_qwen_wording_says_the_pictures_are_yours_without_warnings():
    """Qwen confirmed outputs are the user's (2026-09-21); the record says so plainly."""
    from image_to_3dlab.provenance import QWEN_OUTPUT_RIGHTS

    assert "are yours" in QWEN_OUTPUT_RIGHTS
    assert "ambiguous" not in QWEN_OUTPUT_RIGHTS and "non-commercial" not in QWEN_OUTPUT_RIGHTS.lower()
