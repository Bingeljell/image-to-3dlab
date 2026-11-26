"""Check that the preview command passes both independent geometry options."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import blender_render_asset


@pytest.mark.parametrize(
    ("options", "culled", "recalculate"),
    [
        ([], False, False),
        (["--culled"], True, False),
        (["--recalc-normals"], False, True),
        (["--culled", "--recalc-normals"], True, True),
    ],
)
def test_preview_command_passes_geometry_options(
    options, culled, recalculate, monkeypatch, tmp_path, capsys
):
    asset, output = tmp_path / "model.glb", tmp_path / "renders"

    def bundle(worker, arguments):
        assert worker.name == "blender_render_asset_worker.py"
        assert arguments["culled"] is culled
        assert arguments["recalc_normals"] is recalculate
        assert arguments["world_color"] == (0.22, 0.19, 0.15)
        assert arguments["raytracing"] is True
        return "bundled"

    def sendall(message):
        assert output.is_dir()
        assert json.loads(message) == {
            "type": "execute_code",
            "params": {"code": "bundled"},
        }

    class Connection:
        def __enter__(self):
            return SimpleNamespace(
                sendall=sendall,
                settimeout=lambda _: None,
                recv=lambda _: b'{"status": "success"}',
            )

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(blender_render_asset, "build_worker_code", bundle)
    monkeypatch.setattr(
        blender_render_asset.socket,
        "create_connection",
        lambda *_args, **_kwargs: Connection(),
    )
    monkeypatch.setattr(
        sys, "argv", ["preview", str(asset), str(output), "--env", "studio", *options]
    )
    assert blender_render_asset.main() == 0
    assert capsys.readouterr().out.strip() == '{"status": "success"}'
