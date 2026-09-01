from __future__ import annotations

import pytest

from image_to_3dlab.rig_profile import RigProfileError, load_rig_profile


def test_shipped_quadruped_profile_is_valid_and_connected():
    profile = load_rig_profile("rigify.basic-quadruped.blender-5.2.v1")

    assert profile["id"] == "rigify.basic-quadruped.blender-5.2.v1"
    assert profile["semanticProfile"] == "rigify.quadruped.v1"
    assert profile["blenderVersion"] == "5.2"
    assert profile["joints"]["front.left.elbow"]["targets"] == [
        {"bone": "front_thigh.L", "endpoint": "tail"},
        {"bone": "front_shin.L", "endpoint": "head"},
    ]
    assert profile["joints"]["front.left.elbow"]["mirrorOf"] == "front.right.elbow"
    assert profile["joints"]["back.right.knee"]["sourceBone"] == "DEF-thigh.R"


@pytest.mark.parametrize("profile_id", ["../secret", "a/b", "", "profile id"])
def test_profile_ids_cannot_escape_the_profile_directory(profile_id):
    with pytest.raises(RigProfileError):
        load_rig_profile(profile_id)
