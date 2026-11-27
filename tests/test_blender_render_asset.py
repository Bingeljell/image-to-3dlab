"""Check the package entry point's live preview payload."""

from pathlib import Path

from scripts import blender_render_asset


def test_preview_packages_worker_and_remote_paths(monkeypatch):
    asset = Path("/remote assets/artist's model.glb")
    output_dir = Path("/remote renders/cardinal")

    def bundle(path, arguments):
        assert path.name == "blender_render_asset_worker.py" and path.is_file()
        assert arguments == {
            "asset_path": str(asset),
            "output_dir": str(output_dir),
            "label": "model",
            "culled": False,
            "recalc_normals": False,
            "world_color": (0.035, 0.035, 0.035),
            "raytracing": False,
        }
        return "bundled"

    monkeypatch.setattr(blender_render_asset, "build_worker_code", bundle)
    assert (
        blender_render_asset.blender_code(asset, output_dir, "model", "dark")
        == "bundled"
    )
