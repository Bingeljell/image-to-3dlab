"""Validate add-on documents and arguments before any tool can start."""

import json

import pytest

from image_to_3dlab import addons


def manifest(**overrides):
    return {
        "schema": 1, "id": "test", "name": "Test", "version": "1.0", "author": {"name": "A"},
        "licence": "MIT", "summary": "A test model tool",
    } | overrides


def listing(**overrides):
    return {
        "id": "test", "name": "Test", "repo": "https://example.invalid/test",
        "commit": "abc", "author": {"name": "A"},
    } | overrides


@pytest.mark.parametrize(("source", "error"), [
    (manifest(tools=[{"id": "run", "label": "Run", "hint": "", "run": ["demo"], "makes": "glb",
                      "inputs": {"bad": "type"}}]), "inputs must be a list"),
    (manifest(schema=2), "schema must be 1"),
    (manifest(type="ignored"), "'type' must be one of"),
    (manifest(runs="ignored"), "'runs' must be one of"),
    (manifest(author="A"), "author.*must be an object"),
    (manifest(author={}), "author.*missing 'name'"),
])
def test_manifest_rejects_invalid_declarations(tmp_path, source, error):
    path = tmp_path / "addon.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match=error):
        addons.read_manifest(path)


@pytest.mark.parametrize(("data", "error"), [
    ({}, "registry must be a list"),
    ([None], "must be an object"),
    ([listing(commit="")], "pinned commit"),
    ([listing(type="test")], "'type' must be one of"),
    ([listing(tier="unknown")], "'tier' must be one of"),
])
def test_registry_refuses_invalid_listing(data, error):
    with pytest.raises(ValueError, match=error):
        addons.read_registry(data)


def test_registry_drops_malformed_optional_metadata_and_keeps_partial_result():
    result, partial, incomplete = addons.read_registry([
        listing(rating="unrated", ratings_count="n/a", links=[],
                picture="http://example.invalid/picture", run_check={"status": "other"}),
        listing(run_check={"status": "partial", "skipped": ["tool", False, "other"]}),
        listing(run_check={"status": "partial"}),
    ])
    assert (result.rating, result.ratings_count, result.links, result.picture, result.run_check) == (
        None, 0, {}, None, "",
    )
    assert partial.run_check == "partial" and partial.run_skipped == ("tool", "other")
    assert incomplete.run_skipped == ()
    assert "(tool, other)" in addons.tier_sentence("listed", partial.run_check, partial.run_skipped)


@pytest.fixture
def all_inputs():
    return addons.tool_from_dict({
        "id": "make", "label": "Make", "hint": "Make a model", "makes": "glb",
        "run": ["demo", "--count", "{count}", "--shape", "{shape}", "--out", "{out}"],
        "inputs": [
            {"id": "count", "type": "number", "label": "Count", "min": 0, "max": 3},
            {"id": "shape", "type": "choice", "label": "Shape", "choices": ["cube", "sphere"]},
            {"id": "merge", "type": "toggle", "label": "Merge", "optional": True},
            {"id": "name", "type": "text", "label": "Name", "optional": True},
            {"id": "source", "type": "picture", "label": "Source", "optional": True},
        ],
    })


@pytest.mark.parametrize(("changes", "error"), [
    ({"count": True}, "must be a number"),
    ({"count": -1}, "must be at least 0"),
    ({"count": 4}, "must be at most 3"),
    ({"shape": "cone"}, "must be one of"),
    ({"merge": "yes"}, "must be true or false"),
    ({"name": 42}, "must be text"),
])
def test_command_build_rejects_bad_values_from_api(all_inputs, changes, error):
    with pytest.raises(ValueError, match=error):
        addons.build_command(all_inputs, {"count": 2, "shape": "cube"} | changes, out_dir="out", blender="blender")


@pytest.mark.parametrize(("raw", "error"), [
    ({"count": "twelve"}, "must be a number"),
    ({"merge": "maybe"}, "must be true or false"),
    ({"source": "no.png"}, "file does not exist"),
])
def test_cli_reports_bad_addon_arguments(all_inputs, tmp_path, raw, error):
    with pytest.raises(ValueError, match=error):
        addons.parse_inputs(raw, all_inputs, cwd=tmp_path)


@pytest.mark.parametrize(("toggle", "expected"), [("yes", True), ("no", False)])
def test_addon_cli_coerces_inputs_and_builds_literal_arguments(all_inputs, tmp_path, toggle, expected):
    path = tmp_path / "picture with spaces.png"
    path.touch()
    inputs = addons.parse_inputs({"count": "2.5", "shape": "sphere", "merge": toggle,
                                  "name": "small prop", "source": str(path)}, all_inputs)
    assert inputs == {"count": 2.5, "shape": "sphere", "merge": expected,
                      "name": "small prop", "source": str(path)}
    assert addons.build_command(all_inputs, inputs, out_dir=tmp_path, blender="blender") == [
        "demo", "--count", "2.5", "--shape", "sphere", "--out", str(tmp_path),
    ]


def test_optional_input_cannot_be_omitted_when_command_references_it(all_inputs):
    from dataclasses import replace

    tool = replace(all_inputs, run=(*all_inputs.run, "{merge}"))
    with pytest.raises(ValueError, match="missing required input: merge"):
        addons.build_command(tool, {"count": 2, "shape": "cube"}, out_dir="out", blender="blender")
