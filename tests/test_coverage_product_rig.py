"""Reject malformed rig corrections before they can reach the scene adapter."""

import pytest
from test_rig_sidecar import sidecar_for, with_binding

from image_to_3dlab import rig_sidecar as rig


@pytest.mark.parametrize(("fields", "value", "message"), [
    (("mirror", "axis"), "A", "mirror.axis must be"),
    (("mirror", "origin"), True, "mirror.origin must be finite"),
    (("mirror", "origin"), float("inf"), "mirror.origin must be finite"),
    (("joints",), {}, "joints must contain"),
    (("joints", "chest", "position"), [0, 1], "exactly three"),
    (("joints", "chest", "sourceBone"), " ", "non-empty string"),
    (("joints", "chest"), None, "must be an object"),
    (("joints", "chest", "added"), 3, "unknown fields: added"),
    (("corrections", "absent"), {}, "references unknown joint"),
    (("corrections", "shoulder.L", "mirrored"), "false", "must be boolean"),
    (("assetFingerprint",), "sha256:abc", "64 lowercase"),
    (("assetFingerprint",), "sha256:" + "Z" * 64, "64 lowercase"),
    (("binding", "extra"), 1, "binding fields do not match"),
    (("binding", "joints", "absent"), {}, "references unknown joint"),
    (("binding", "joints", "shoulder.L", "extra"), 1, "fields do not match"),
    (("binding", "joints", "shoulder.L", "targets"), [], "at least one target"),
    (("binding", "joints", "shoulder.L", "targets", 0), {}, "fields do not match"),
    (("binding", "joints", "shoulder.L", "targets", 0, "endpoint"), "center", "must be head or tail"),
])
def test_rig_sidecar_rejects_invalid_field(fields, value, message):
    source = with_binding(sidecar_for(b"glb"), b"scene")
    section = source
    for name in fields[:-1]:
        section = section[name]
    section[fields[-1]] = value
    with pytest.raises(rig.RigSidecarError, match=message):
        rig.validate_sidecar(source)


@pytest.mark.parametrize("contents", [None, "{invalid"])
def test_rig_sidecar_reports_read_error(tmp_path, contents):
    path = tmp_path / "rig.json"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(rig.RigSidecarError, match="cannot read rig sidecar"):
        rig.load_sidecar(path)


def profile_source():
    import json

    from image_to_3dlab.rig_profile import PROFILE_ROOT

    return json.loads((PROFILE_ROOT / "rigify.basic-quadruped.blender-5.2.v1.json").read_text())


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda source: source.clear(), "root fields"),
    (lambda source: source.update(schemaVersion=2), "schemaVersion"),
    (lambda source: source.update(id=""), "id must be"),
    (lambda source: source.update(blenderMetarigNames=[]), "non-empty array"),
    (lambda source: source.update(joints=[]), "non-empty object"),
    (lambda source: source.update(joints={"chest": {}}), "fields do not match"),
    (lambda source: next(iter(source["joints"].values())).update(targets=[]), "at least one"),
    (lambda source: next(iter(source["joints"].values())).update(targets=[{}]), "invalid Blender target"),
    (lambda source: next(iter(source["joints"].values())).update(targets=[{"bone": "spine", "endpoint": "middle"}]),
     "endpoint must be"),
    (lambda source: next(iter(source["joints"].values())).update(parent="missing"), "unknown joint"),
])
def test_profile_catches_schema_errors_at_load(tmp_path, mutate, message):
    import json

    from image_to_3dlab.rig_profile import RigProfileError, load_rig_profile

    source = profile_source()
    profile_id = source["id"]
    mutate(source)
    (tmp_path / f"{profile_id}.json").write_text(json.dumps(source))
    with pytest.raises(RigProfileError, match=message):
        load_rig_profile(profile_id, tmp_path)


@pytest.mark.parametrize(("profile_id", "message"), [
    ("", "unsafe characters"), ("../other", "unsafe characters"), ("missing", "cannot load rig profile"),
])
def test_profile_loader_rejects_invalid_filename(tmp_path, profile_id, message):
    from image_to_3dlab.rig_profile import RigProfileError, load_rig_profile

    with pytest.raises(RigProfileError, match=message):
        load_rig_profile(profile_id, tmp_path)


def test_profile_must_match_its_filename_and_have_an_object_root(tmp_path):
    import json

    from image_to_3dlab.rig_profile import (
        RigProfileError,
        load_rig_profile,
        validate_rig_profile,
    )

    (tmp_path / "other.json").write_text(json.dumps(profile_source()))
    with pytest.raises(RigProfileError, match="expected 'other'"):
        load_rig_profile("other", tmp_path)
    with pytest.raises(RigProfileError, match="root must be an object"):
        validate_rig_profile([])
