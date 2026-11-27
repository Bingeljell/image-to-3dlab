"""Exercise CLI reporting of open mesh boundary loops."""

import json
import runpy
import sys

import numpy as np
import pytest
import trimesh

from scripts import tear_provenance


@pytest.mark.parametrize("record_json", [False, True])
def test_tear_cli_reports_open_box_with_optional_json(monkeypatch, tmp_path, capsys, record_json):
    mesh = trimesh.creation.box()
    mesh.update_faces(mesh.face_normals[:, 2] < 1)
    source, out = tmp_path / "box.glb", tmp_path / "sub" / "tears.json"
    mesh.export(source)
    args = ["tear_provenance.py", str(source), "--view", "0", "0", "1", "--min-perimeter", "1"]
    if record_json:
        args.extend(["--json", str(out)])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as status:
        runpy.run_path(tear_provenance.__file__, run_name="__main__")
    assert status.value.code == 0
    printed = capsys.readouterr().out
    assert "boundary loops: 1  large: 1" in printed
    assert "4.000" in printed
    assert out.exists() == record_json
    if record_json:
        report = json.loads(out.read_text())
        tear = report["tears"][0]
        assert tear["perimeter_rel"] == pytest.approx(4)
        np.testing.assert_allclose(report["view"], [0, 0, 1])


def test_tear_report_labels_the_three_distinct_states(capsys):
    result = {
        "loops_total": 3, "loops_large": 3, "view": [0, 0, 1],
        "summary": {"large_tears": 3, "facing_away": 1, "occluded": 1,
                    "visible_to_input_view": 1, "visible_perimeter_share": 0.25},
        "tears": [
            {"centroid_rel": [0, 0, 0], "perimeter_rel": perimeter, "facing_dot": 1 if faces else -1,
             "was_visible": visible, "facing_camera": faces}
            for perimeter, visible, faces in [(4, True, True), (3, False, True), (2, False, False)]
        ],
    }
    tear_provenance.report(result)
    output = capsys.readouterr().out
    assert "25% of total tear perimeter" in output
    assert output.rstrip().splitlines()[-3].endswith("VISIBLE")
    assert output.rstrip().splitlines()[-2].endswith("occluded")
    assert output.rstrip().splitlines()[-1].endswith("facing away")
