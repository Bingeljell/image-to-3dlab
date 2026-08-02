import pytest

from image_to_3dlab.provenance import validate_run_policy


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
