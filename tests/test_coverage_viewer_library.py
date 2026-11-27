"""Library and Finish handle moved, missing, and partially written run records."""

import hashlib
import json

import assets_api
import finish_api
import pytest


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_library_matches_finish_inputs_by_hash_when_model_has_unrelated_name(tmp_path, monkeypatch):
    picture_name = "a-particularly-long-picture-title-for-the-library__20260101-080001"
    picture = _write(tmp_path / "images" / picture_name / f"{picture_name}.png", b"picture bytes")
    record = picture.with_suffix(".provenance.json")
    record.write_text('{"settings": {"seed": true, "steps": 8}}', encoding="utf-8")
    model = _write(tmp_path / "different-name" / "different-name.glb", b"model bytes")
    finish = tmp_path / "finish" / "run__finish__20260101-080005"
    result = _write(finish / "object_20k.glb", b"finished bytes")
    _write(finish / "input" / "source.png", picture.read_bytes())
    _write(finish / "input" / "source.glb", model.read_bytes())
    for area in ("finish", "animate", "props"):
        _write(tmp_path / area / "invalid run name" / "stray.glb", b"must not list")
    _write(tmp_path / "finish" / "empty__finish__20260101-080006" / "steps" / "placeholder", b"partial")
    # Saving the cache may fail independently of reading all the runs.
    (tmp_path / assets_api.INDEX_NAME).mkdir()
    monkeypatch.setattr(assets_api, "RECIPES", tmp_path / "missing-prompts.json")
    [asset] = assets_api.list_assets(tmp_path)
    assert asset["picture"] == picture.relative_to(tmp_path).as_posix()
    assert asset["model"] == model.relative_to(tmp_path).as_posix()
    assert asset["finished"] == result.relative_to(tmp_path).as_posix()
    assert asset["picture_settings"] is None
    assert asset["name"] == "run"
    assert assets_api.friendly_name(picture_name).endswith("…")


def test_library_ignores_props_without_a_lod_or_source(tmp_path):
    run = tmp_path / "props" / "crate__props__20260101-080005"
    _write(run / "finished" / "crate" / "crate_LOD1.glb", b"no first lod")
    [asset] = assets_api.list_assets(tmp_path)
    assert asset["props"] == []
    assert asset["stage"] == "model"


def test_clearing_a_model_main_flag_does_not_clear_the_family_other_main(tmp_path):
    picture = _write(tmp_path / "images" / "picture" / "picture.png", b"picture")
    for route in ("trellis", "sf3d"):
        name = f"{picture.stem}__{route}"
        _write(tmp_path / "commercial" / name / f"{name}.glb", route.encode())
    first, second = assets_api.list_assets(tmp_path)
    assets_api.set_main(tmp_path, first["id"], True)
    assets_api.set_main(tmp_path, second["id"], False)
    assert [item["id"] for item in assets_api.list_assets(tmp_path) if item["main"]] == [first["id"]]
    assets_api.set_main(tmp_path, first["id"], False)
    assert not any(item["main"] for item in assets_api.list_assets(tmp_path))
    with pytest.raises(ValueError, match="no such asset"):
        assets_api.set_name(tmp_path, "a" * 12, "Gone")


def test_source_files_are_found_beside_record_after_run_folder_moves(tmp_path):
    asset = b"mesh"
    record = {
        "backend": "pixal3d", "license": {"name": "fixture"},
        "output": {"path": str(tmp_path / "gone" / "model.glb"), "sha256": hashlib.sha256(asset).hexdigest()},
        "post_dump": "model.post.bin",
    }
    run = tmp_path / "current"
    record_path = _write(run / "model.provenance.json", json.dumps(record).encode())
    assert finish_api.find_source_views(asset, root=tmp_path) is None
    assert finish_api.find_source_dump(asset, root=tmp_path) is None
    views = _write(run / "model.svviews" / "transforms.json", b"{}").parent
    dump = _write(run / "model.post.bin", b"mesh dump")
    assert finish_api.find_source_views(asset, root=tmp_path) == views
    assert finish_api.find_source_dump(asset, root=tmp_path) == dump
    assert finish_api.find_source_record(asset, root=tmp_path) == record_path


@pytest.mark.symlinks
def test_source_record_lookup_ignores_link_leaving_output(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    record = _write(tmp_path / "outside.json", json.dumps({
        "license": {"name": "fixture"}, "output": {"sha256": hashlib.sha256(b"mesh").hexdigest()},
    }).encode())
    (root / "linked.json").symlink_to(record)
    assert finish_api.find_source_record(b"mesh", root=root) is None
