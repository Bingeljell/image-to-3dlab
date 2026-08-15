"""Tests for the decode cache and the re-bake argument mapping.

Two things are pinned here because both have already gone wrong in this repo:

* **`decimation_target` is vertices, not faces.** The port's flag is `--bake-target-faces`
  and feeds a face count into a parameter whose docstring says "target number of vertices",
  and that unit error propagated into the manifests. The re-bake script must pass the value
  through unchanged and be named honestly.
* **The patch's anchors.** Patch scripts here have historically assumed anchors that moved,
  applied twice, or been impossible to undo — so state/apply/remove are tested directly.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebake = _load("trellis_rebake", "trellis_rebake.py")
patch = _load("patch_dump_decode", "patch_trellis_dump_decode.py")


# --- the to_glb argument mapping ---------------------------------------------------

def _args(**kw):
    defaults = {
        "decimation_target": 500_000, "texture_size": 3072,
        "remesh": False, "remesh_band": 1.0, "remesh_project": 0.0,
    }
    return argparse.Namespace(**{**defaults, **kw})


PAYLOAD = {
    "vertices": "V", "faces": "F", "attrs": "A",
    "coords": "C", "layout": "L", "voxel_size": 0.5,
}


def test_maps_every_payload_field_to_its_to_glb_name():
    """attrs -> attr_volume and layout -> attr_layout are easy to get silently wrong."""
    kwargs = rebake.build_to_glb_kwargs(PAYLOAD, _args())
    assert kwargs["vertices"] == "V"
    assert kwargs["faces"] == "F"
    assert kwargs["attr_volume"] == "A"
    assert kwargs["attr_layout"] == "L"
    assert kwargs["coords"] == "C"
    assert kwargs["voxel_size"] == 0.5


def test_decimation_target_passes_through_unconverted():
    """It is a VERTEX budget. Any face/vertex arithmetic here would repeat the old bug."""
    assert rebake.build_to_glb_kwargs(PAYLOAD, _args(decimation_target=500_000))[
        "decimation_target"
    ] == 500_000


def test_remesh_defaults_off_but_is_expressible():
    assert rebake.build_to_glb_kwargs(PAYLOAD, _args())["remesh"] is False
    assert rebake.build_to_glb_kwargs(PAYLOAD, _args(remesh=True))["remesh"] is True


def test_reference_branch_settings_match_the_demo():
    """app.py:503-505 hardcodes remesh_band=1, remesh_project=0. Ours must agree."""
    kwargs = rebake.build_to_glb_kwargs(PAYLOAD, _args(remesh=True))
    assert kwargs["remesh_band"] == 1.0
    assert kwargs["remesh_project"] == 0.0


def test_summarise_reports_counts():
    class Shaped:
        def __init__(self, n):
            self.shape = (n, 3)

    text = rebake.summarise({"vertices": Shaped(1978486), "faces": Shaped(3999999)})
    assert "1,978,486 vertices" in text
    assert "3,999,999 faces" in text


def test_split_material_cache_reuses_geometry_without_overwriting_new_attrs(tmp_path):
    class FakeTorch:
        payloads = {}

        @classmethod
        def load(cls, path, weights_only=False):
            return cls.payloads[Path(path)]

    geometry = tmp_path / "geometry.pt"
    material = tmp_path / "candidate.pt"
    geometry.touch()
    material.touch()
    FakeTorch.payloads = {
        geometry: {"vertices": "V", "faces": "F", "attrs": "old", "coords": "oldC"},
        material: {"geometry_ref": str(geometry), "attrs": "new", "coords": "newC"},
    }

    payload = rebake.load_payload(material, FakeTorch)
    assert payload["vertices"] == "V"
    assert payload["faces"] == "F"
    assert payload["attrs"] == "new"
    assert payload["coords"] == "newC"


def test_split_material_cache_reports_missing_geometry(tmp_path):
    class FakeTorch:
        @staticmethod
        def load(path, weights_only=False):
            return {"geometry_ref": "missing.pt"}

    material = tmp_path / "candidate.pt"
    material.touch()
    with pytest.raises(FileNotFoundError, match="geometry_ref"):
        rebake.load_payload(material, FakeTorch)


def test_existing_glb_geometry_replaces_only_vertices_and_faces():
    import torch

    class Mesh:
        vertices = torch.tensor([[1.0, 2.0, 3.0]]).numpy()
        faces = torch.tensor([[0, 0, 0]]).numpy()

        def merge_vertices(self, merge_tex, merge_norm):
            assert merge_tex is True
            assert merge_norm is True

    class Trimesh:
        @staticmethod
        def load(path, force, process):
            assert path == "accepted.glb"
            assert force == "mesh"
            assert process is False
            return Mesh()

    payload = {
        "vertices": torch.zeros((2, 3)),
        "faces": torch.zeros((2, 3)),
        "attrs": "keep",
    }
    replaced = rebake.replace_geometry_from_glb(
        payload, Path("accepted.glb"), torch, Trimesh
    )
    assert replaced["vertices"].shape == (1, 3)
    assert replaced["vertices"].tolist() == [[1.0, -3.0, 2.0]]
    assert replaced["faces"].dtype == torch.int32
    assert replaced["attrs"] == "keep"


# --- the generate.py patch ---------------------------------------------------------

GENERATE = '''\
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.9)

    mesh_out = outputs[0] if isinstance(outputs, list) else outputs
    t_gen = time.time() - t0
'''


def test_detects_absent_then_applied():
    assert patch.state(GENERATE) == "absent"
    assert patch.state(patch.apply(GENERATE)) == "applied"


def test_apply_is_reversible():
    assert patch.remove(patch.apply(GENERATE)) == GENERATE


def test_patched_source_is_valid_python_in_context():
    import ast

    body = patch.apply(GENERATE)
    ast.parse("def main():\n" + "".join("    " + ln + "\n" for ln in body.splitlines()))


def test_half_patched_source_is_refused():
    """A partial apply must not look like a clean state in either direction."""
    half = GENERATE.replace(patch.ANCHOR_ARG, patch.ANCHOR_ARG + patch.ADDED_ARG, 1)
    with pytest.raises(RuntimeError, match="half-patched"):
        patch.state(half)


def test_missing_anchor_raises():
    with pytest.raises(RuntimeError, match="anchor missing"):
        patch.state("def main():\n    pass\n")


def test_dump_imports_what_it_uses():
    """Patched code must not assume the host file's imports.

    A missing one-line import once cost a full 16-minute run that crashed on its last
    statement, which is why this is asserted rather than eyeballed.
    """
    assert "import torch as _torch" in patch.ADDED_DUMP
    assert "_torch.save" in patch.ADDED_DUMP
