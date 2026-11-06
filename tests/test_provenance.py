import ast
import json
from pathlib import Path

import pytest
from glb_fixtures import read_glb, valid_glb

from image_to_3dlab import __version__
from image_to_3dlab.provenance import (
    LICENSES,
    finalize_output,
    set_glb_generator,
    validate_run_policy,
    write_provenance,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sf3d_is_allowed_when_conditionals_are_enabled():
    profile = validate_run_policy(
        "sf3d", "game", "worldwide", True, model_identity="stabilityai/stable-fast-3d"
    )
    assert profile.classification == "commercial-conditional"


def test_hunyuan_is_blocked_for_worldwide_game():
    with pytest.raises(ValueError, match="worldwide"):
        validate_run_policy("hunyuan-comfyui", "game", "worldwide", True)


def test_trellis_inherits_dinov3_conditional_classification():
    profile = validate_run_policy(
        "trellis2", "game", "worldwide", True, model_identity="microsoft/TRELLIS.2-4B"
    )
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
    assert validate_run_policy(
        "qwen-image-2.1", "game", "worldwide", allow_conditional=True,
        model_identity="Qwen/Qwen-Image-2.1",
    ) is profile
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
    profile = validate_run_policy(
        research_model, use_case="showcase", distribution="private",
        allow_conditional=True, model_identity="Qwen/Qwen-Image-2.1",
    )
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


def test_validate_run_policy_accepts_canonical_sf3d_identity():
    profile = validate_run_policy(
        "sf3d", "game", "worldwide", True, model_identity="stabilityai/stable-fast-3d"
    )
    assert profile is LICENSES["sf3d"]


def test_validate_run_policy_rejects_unknown_identity():
    with pytest.raises(ValueError, match="someone/unknown-repo"):
        validate_run_policy(
            "sf3d", "game", "worldwide", True, model_identity="someone/unknown-repo"
        )


def test_missing_identity_is_refused_unless_unverified_is_allowed():
    with pytest.raises(ValueError, match="no reviewed model identity"):
        validate_run_policy(
            "hunyuan-comfyui", "showcase", "private", True, model_identity=None
        )
    profile = validate_run_policy(
        "hunyuan-comfyui", "showcase", "private", True,
        model_identity=None, allow_unverified=True,
    )
    assert profile is LICENSES["hunyuan-comfyui"]


def test_commercial_intent_refuses_a_conditional_backend():
    with pytest.raises(ValueError, match="commercial_intent"):
        validate_run_policy(
            "sf3d", "showcase", "private", True,
            model_identity="stabilityai/stable-fast-3d",
            commercial_intent=True,
        )


def test_allow_research_only_false_refuses_a_private_showcase(research_model):
    with pytest.raises(ValueError, match="allow_research_only"):
        validate_run_policy(
            research_model, use_case="showcase", distribution="private",
            allow_conditional=True,
            model_identity="Qwen/Qwen-Image-2.1",
            allow_research_only=False,
        )


def test_finalize_output_does_not_give_a_missing_identity_the_default_licence(tmp_path):
    image = tmp_path / "input.png"
    generated = tmp_path / "out.glb"
    image.write_bytes(b"image")
    generated.write_bytes(valid_glb(generator="Khronos glTF Blender I/O v5.2.39"))
    profile = LICENSES["hunyuan-comfyui"]

    _, sidecar = finalize_output(
        generated=generated,
        image=image,
        output_root=tmp_path,
        backend="hunyuan-comfyui",
        profile=profile,
        intent={"use_case": "showcase", "distribution": "private", "commercial_intent": False},
        parameters={},
        source_manifest=None,
        model_identity=None,
    )

    record = json.loads(sidecar.read_text())
    assert record["output"]["classification"] == "unverified"
    assert record["license"]["verified"] is False
    assert "Tencent" not in record["license"]["name"]


def test_validate_run_policy_allows_unknown_identity_when_unverified_allowed():
    profile = validate_run_policy(
        "sf3d", "game", "worldwide", True, model_identity="someone/unknown-repo",
        allow_unverified=True,
    )
    assert profile is LICENSES["sf3d"]


def test_finalize_output_records_canonical_identity(tmp_path):
    image = tmp_path / "input.png"
    generated = tmp_path / "out.glb"
    image.write_bytes(b"image")
    generated.write_bytes(valid_glb(generator="Khronos glTF Blender I/O v5.2.39"))
    profile = LICENSES["sf3d"]

    destination, sidecar = finalize_output(
        generated=generated,
        image=image,
        output_root=tmp_path,
        backend="sf3d",
        profile=profile,
        intent={"use_case": "showcase", "distribution": "private", "commercial_intent": False},
        parameters={"model": "stabilityai/stable-fast-3d"},
        source_manifest=None,
        model_identity="stabilityai/stable-fast-3d",
    )

    record = json.loads(sidecar.read_text())
    assert record["model"]["identity"] == "stabilityai/stable-fast-3d"
    assert record["output"]["classification"] == profile.classification
    assert "verified" not in record.get("license", {})
    assert record["made_with"]["name"] == "AssetFurnace"
    assert record["made_with"]["version"] == __version__
    assert read_glb(destination.read_bytes())[0]["asset"]["generator"] == f"AssetFurnace {__version__}"


def test_finalize_output_marks_unknown_identity_unverified(tmp_path):
    image = tmp_path / "input.png"
    generated = tmp_path / "out.glb"
    image.write_bytes(b"image")
    generated.write_bytes(valid_glb(generator="Khronos glTF Blender I/O v5.2.39"))
    profile = LICENSES["sf3d"]

    _, sidecar = finalize_output(
        generated=generated,
        image=image,
        output_root=tmp_path,
        backend="sf3d",
        profile=profile,
        intent={"use_case": "showcase", "distribution": "private", "commercial_intent": False},
        parameters={"model": "someone/unknown-repo"},
        source_manifest=None,
        model_identity="someone/unknown-repo",
    )

    record = json.loads(sidecar.read_text())
    assert record["model"]["identity"] == "someone/unknown-repo"
    assert record["output"]["classification"] == "unverified"
    assert record["license"]["name"] == "Unverified — user-declared model"
    assert record["license"]["verified"] is False
    assert "url" not in record["license"]
    assert record["license"]["conditions"] == [
        "Verify this model's licence yourself; this run used an identity not in the reviewed profile."
    ]


# --------------------------------------------------------------- who made the file

def test_every_record_says_who_made_the_file(tmp_path):
    record = {"schema_version": 1, "license": {"name": "Apache-2.0", "conditions": ["credit"]}}
    path = write_provenance(tmp_path / "a.provenance.json", record)
    saved = json.loads(path.read_text())
    assert saved["made_with"] == {
        "name": "AssetFurnace",
        "version": __version__,
        "url": "https://assetfurnace.com",
    }
    assert saved["schema_version"] == 1, "the schema version does not move"
    assert saved["license"] == {"name": "Apache-2.0", "conditions": ["credit"]}
    assert "made_with" not in saved["license"], "who made it is never part of the licence"


def test_an_old_record_without_made_with_still_loads(tmp_path):
    old = {"schema_version": 1, "license": {"name": "MIT"},
           "output": {"classification": "commercial-clear"}}
    path = tmp_path / "old.provenance.json"
    path.write_text(json.dumps(old))
    assert json.loads(path.read_text()) == old


def test_a_glb_says_it_was_made_with_assetfurnace_and_nothing_else_changes(tmp_path):
    path = tmp_path / "a.glb"
    before_bytes = valid_glb(generator="Khronos glTF Blender I/O v5.2.39",
                             asset_extra={"minVersion": "2.0"}, binary=b"\x01\x02\x03\x04")
    path.write_bytes(before_bytes)
    before, before_bin = read_glb(before_bytes)

    set_glb_generator(path)

    after, after_bin = read_glb(path.read_bytes())
    expected = json.loads(json.dumps(before))
    expected["asset"]["generator"] = f"AssetFurnace {__version__}"
    assert after == expected, "only generator changes"
    assert after_bin == before_bin, "the binary chunk is untouched"
    set_glb_generator(path)
    assert read_glb(path.read_bytes())[0]["asset"]["generator"] == f"AssetFurnace {__version__}"


def test_a_file_that_is_not_a_glb_is_refused_not_rewritten(tmp_path):
    path = tmp_path / "not.glb"
    path.write_bytes(b"glb")
    with pytest.raises(ValueError, match="not a GLB"):
        set_glb_generator(path)
    assert path.read_bytes() == b"glb"


def _provenance_writers() -> list[str]:
    """Every write call that targets a provenance sidecar, by file and line."""
    offenders = []
    for folder in ("image_to_3dlab", "viewer", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr not in {"write_text", "write"}:
                    continue
                segment = ast.get_source_segment(source, node) or ""
                receiver = ast.get_source_segment(source, func.value) or ""
                if not ("provenance" in (segment + receiver).lower()
                        or "sidecar" in receiver.lower()):
                    continue
                line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                if "write_provenance" in segment or "not a provenance record" in line:
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return offenders


def test_every_provenance_writer_uses_the_shared_helper():
    """A new writer that forgets to stamp the record fails here, before it ships."""
    assert _provenance_writers() == []


def test_the_scan_would_catch_a_writer_that_forgets(tmp_path):
    """The guard is only useful if it really flags a raw sidecar write."""
    forgetful = tmp_path / "example.py"
    forgetful.write_text(
        'from pathlib import Path\n'
        'def write(sidecar: Path, record):\n'
        '    sidecar.write_text("{}")\n'
    )
    tree = ast.parse(forgetful.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any("sidecar" in (ast.get_source_segment(forgetful.read_text(), call) or "")
               for call in calls)
@pytest.mark.parametrize("extra", [False, True])
def test_derived_sidecar_retains_licences_and_hashes(tmp_path, extra):
    from image_to_3dlab import provenance as p

    source, output = tmp_path / "source.glb", tmp_path / "out.glb"
    source.write_bytes(b"source")
    output.write_bytes(b"derived")
    origin = {"license": {"name": "Shape", "conditions": ["credit shape"]},
              "output": {"classification": "commercial-conditional"}, "model": {"identity": "shape"},
              "components": [{"component": "shape encoder"}]}
    paint = {"license": {"name": "Paint", "url": "https://example.test/terms", "conditions": ["territory"]},
             "output": {"classification": "territory-restricted"},
             "components": [{"component": "paint"}]}
    source.with_suffix(".json").write_text(json.dumps(origin))
    sidecar = p.write_derived_sidecar(source, output, operation="finish", component_records=[paint] if extra else [])
    record = json.loads(sidecar.read_text())
    assert record["source_record"] == origin
    assert record["made_with"]["name"] == "AssetFurnace"
    assert record["output"]["sha256"] == p.sha256_file(output)
    assert record["model"] == origin["model"]
    assert record["components"][0] == origin["components"][0]
    assert "credit shape" in record["license"]["conditions"]
    if extra:
        assert "territory" in record["license"]["conditions"]
        assert record["component_records"] == [paint]
        assert record["output"]["classification"] == "unverified"
    else:
        assert record["license"] == origin["license"]
        assert record["output"]["classification"] == "commercial-conditional"


def test_derived_unknown_source_stays_unverified_and_write_failure_propagates(tmp_path, monkeypatch):
    from pathlib import Path

    from image_to_3dlab import provenance as p

    source, output = tmp_path / "in.glb", tmp_path / "out.glb"
    source.write_bytes(b"source")
    output.write_bytes(b"result")
    record = json.loads(p.write_derived_sidecar(source, output, operation="keep").read_text())
    assert record["output"]["classification"] == "unverified"
    assert record["license"]["verified"] is False
    def fail_replace(*args):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        p.write_derived_sidecar(source, output, operation="keep")
