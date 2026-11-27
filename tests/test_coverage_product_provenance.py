"""Keep license records accurate on malformed and legacy runs."""

import json
import struct
import subprocess

import pytest

from image_to_3dlab import provenance as p


@pytest.mark.parametrize(("data", "message"), [
    (b"short", "not a GLB version 2"),
    (struct.pack("<III", p.GLB_MAGIC, 2, 24) + struct.pack("<II", 4, 0x004E4942) + b"data",
     "no JSON chunk"),
])
def test_invalid_binary_gltf_is_not_rewritten(tmp_path, data, message):
    model = tmp_path / "model.glb"
    model.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        p.set_glb_generator(model)
    assert model.read_bytes() == data


@pytest.mark.parametrize(("data", "message"), [
    (None, "could not read run manifest"), ("[", "could not read run manifest"),
    ("[]", "must be an object"), ('{"schema_version":2}', "schema_version 1"),
])
def test_invalid_manifest_never_returns_data(tmp_path, data, message):
    path = tmp_path / "run.json"
    if data is not None:
        path.write_text(data)
    with pytest.raises(ValueError, match=message):
        p.load_run_manifest(path)


@pytest.mark.parametrize(("backend", "use_case", "distribution", "message"), [
    ("sf3d", "other", "private", "use_case"),
    ("sf3d", "game", "other", "distribution"),
    ("unknown", "game", "private", "no reviewed license"),
])
def test_policy_refuses_unknown_intent_or_backend(backend, use_case, distribution, message):
    with pytest.raises(ValueError, match=message):
        p.validate_run_policy(backend, use_case, distribution, allow_conditional=True)


def test_hunyuan_requires_recording_at_least_one_component(tmp_path):
    with pytest.raises(ValueError, match="name the shape or paint weights"):
        p.write_hunyuan_sidecar(tmp_path / "out.glb", tmp_path / "in.png", backend="hunyuan-shape")


@pytest.mark.parametrize("fault", ["revision", "status"])
def test_repo_revision_distinguishes_unknown_commit_and_dirty_state(monkeypatch, fault):
    def git(argv, **kwargs):
        if fault == "revision" or argv[-1] == "--porcelain":
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0, stdout="abc\n")

    monkeypatch.setattr(p.subprocess, "run", git)
    expected = None if fault == "revision" else {"commit": "abc", "dirty": None}
    assert p._pipeline_revision() == expected


def test_legacy_provenance_ignores_wrong_document_types_and_finds_licence(tmp_path):
    model = tmp_path / "model.glb"
    model.with_suffix(".provenance.json").write_text("[]")
    model.with_suffix(".json").write_text(json.dumps({"old_note": "missing licence"}))
    legacy = {"license": {"name": "MIT"}}
    model.with_suffix(".retopo-repaint.json").write_text(json.dumps(legacy))
    assert p.read_run_record(model) == {"old_note": "missing licence"}
    assert p.read_model_record(model) == legacy
    model.with_suffix(".json").unlink()
    assert p.read_run_record(model) is None
    model.with_suffix(".retopo-repaint.json").unlink()
    assert p.read_model_record(model) is None


def test_derived_licence_keeps_restrictions_from_all_distinct_sources(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(b"source")

    result = tmp_path / "joined.glb"
    result.write_bytes(b"asset")
    first = {"license": {"name": "First", "conditions": ["include attribution", "limit distribution"]},
             "classification": "commercial-clear"}
    other = {"license": {"name": "Second", "conditions": ["include attribution", "keep notice"]},
             "classification": "conditional"}
    out = p.write_derived_sidecar(source, result, operation="join", source_record=first, component_records=[other])
    record = json.loads(out.read_text())
    assert record["license"]["name"] == "First + Second"
    assert record["license"]["conditions"] == ["include attribution", "limit distribution", "keep notice"]
