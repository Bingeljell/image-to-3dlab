"""Test the eye painter command and its socket protocol without Blender."""

import ast
import json
import sys
from runpy import run_path

import pytest

from scripts import paint_eyes


class Connection:
    def __init__(self, reads):
        self.response = iter(reads)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass

    def settimeout(self, timeout):
        self.timeout = timeout

    def sendall(self, data):
        self.request = json.loads(data)

    def recv(self, limit):
        value = next(self.response)
        if isinstance(value, Exception):
            raise value
        assert len(value) <= limit
        return value


@pytest.mark.parametrize(
    "reads,expected",
    [
        ([b'{"status":', b'"ok"}'], '{"status":"ok"}'),
        ([b"partial", TimeoutError()], "partial"),
        ([b"truncated", b""], "truncated"),
    ],
)
def test_socket_stops_at_complete_json_timeout_or_eof(monkeypatch, reads, expected):
    connection = Connection(reads)

    def connect(address, timeout):
        assert address == ("fixture", 1000)
        assert timeout == 10
        return connection

    monkeypatch.setattr(paint_eyes.socket, "create_connection", connect)
    assert paint_eyes.send("print(7)", "fixture", 1000, 23) == expected
    assert connection.timeout == 23
    assert connection.request == {"type": "execute_code", "params": {"code": "print(7)"}}


@pytest.mark.parametrize(
    "options,expected",
    [
        (["--export", "painted.glb"], "export_in_blender"),
        (["--revert"], "revert_in_blender"),
        (["--eye", "-1", "0", "0", "--eye", "1", "0", "0"], "apply_in_blender"),
    ],
)
def test_eye_command_dispatches_to_blender(monkeypatch, capsys, options, expected):
    connection = Connection([b'{"status":"ok"}'])
    monkeypatch.setattr(paint_eyes.socket, "create_connection", lambda *_args, **_kw: connection)
    monkeypatch.setattr(sys, "argv", ["paint_eyes", "--object", "Name with 'quotes'", *options])
    run_path(paint_eyes.__file__, run_name="__main__")
    calls = [node for node in ast.walk(ast.parse(connection.request["params"]["code"])) if isinstance(node, ast.Call)]
    matches = [
        call for call in calls
        if getattr(call.func, "id", getattr(call.func, "attr", None)) == expected
    ]
    assert len(matches) == 1
    assert ast.literal_eval(matches[0].args[0]) == "Name with 'quotes'"
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_eye_cli_passes_style_overrides_and_forward_vector(monkeypatch, capsys):
    connection = Connection([b'{"status":"ok"}'])
    monkeypatch.setattr(paint_eyes.socket, "create_connection", lambda *_args, **_kw: connection)
    monkeypatch.setattr(
        sys, "argv",
        [
            "paint_eyes", "--eye", "0", "0", "0", "--eye", "1", "0", "0",
            "--iris-scale", ".8", "--pupil-scale", ".3", "--iris", "#ffc000",
            "--sclera", ".1,.2,.3", "--pupil=-2,0,255", "--forward", "0", "1", "0",
        ],
    )
    paint_eyes.main()
    payload = ast.parse(connection.request["params"]["code"])
    call = next(node.value for node in payload.body if isinstance(node, ast.Assign))
    assert ast.literal_eval(call.args[-1]) == (0, 1, 0)
    style = ast.literal_eval(call.args[-3].keywords[0].value)
    assert style["iris_scale"] == .8
    assert style["pupil_scale"] == .3
    assert style["sclera_rgb"] == (.1, .2, .3)
    assert style["pupil_rgb"] == (0, 0, 1)
    assert style["iris_inner_rgb"] == pytest.approx((1, 192 / 255, 0))
    assert style["iris_outer_rgb"][0] < style["iris_inner_rgb"][0]
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


@pytest.mark.parametrize(
    "options,error",
    [
        ([], "--eye must be given exactly twice"),
        (["--eye", "1", "0", "0", "--eye", "2", "0", "0", "--iris", "1,2"], "expected three components"),
        (["--eye", "1", "0", "0", "--eye", "2", "0", "0", "--iris", "ff"], "expected #rrggbb"),
    ],
)
def test_eye_cli_rejects_invalid_arguments(monkeypatch, capsys, options, error):
    monkeypatch.setattr(sys, "argv", ["paint_eyes", *options])
    with pytest.raises(SystemExit) as problem:
        paint_eyes.main()
    assert problem.value.code == 2
    assert error in capsys.readouterr().err


@pytest.fixture
def eye_triangles():
    import numpy as np

    centres = np.array([[-2, 0, 0], [2, 0, 0]])
    # The visible triangles face -Y and the last triangle is their back-facing twin.
    triangle = [[-.7, 0, -.7], [.7, 0, -.7], [0, 0, .7]]
    positions = np.array([centres[0] + triangle, centres[1] + triangle, centres[0] + triangle[::-1]])
    coords = np.tile([[.1, .1], [.9, .1], [.5, .9]], (3, 1, 1))
    return positions, coords, centres


@pytest.mark.parametrize("forward", [None, (0, -2, 0)])
def test_bake_eye_patches_selects_front_faces_and_offsets_atlas(eye_triangles, forward):
    import numpy as np

    positions, coords, centres = eye_triangles
    source = np.full((4, 4, 3), .05, dtype=np.float32)
    orm = paint_eyes.constant_orm((4, 4), .65, .75)
    albedo, surface, selection, report = paint_eyes.bake_eye_patches(
        positions, coords, centres, source, orm, 1, 24, 0, paint_eyes.EyeStyle(), forward
    )
    assert albedo.shape == surface.shape == (24, 48, 3)
    assert len(selection) == 2
    assert report["patch"] == [48, 24]
    np.testing.assert_array_equal(selection[0][0], [0])
    np.testing.assert_array_equal(selection[1][0], [1])
    assert (selection[0][1][..., 0] < .5).all()
    np.testing.assert_allclose(selection[1][1] - selection[0][1], [[[.5, 0]] * 3])
    np.testing.assert_allclose(albedo[:, :24], albedo[:, 24:])
    assert surface[12, 12, 1] < surface[0, 0, 1]
    assert surface[12, 12, 2] == pytest.approx(0)
    assert surface[0, 0, 2] == pytest.approx(.75)
    assert [eye["faces"] for eye in report["eyes"]] == [1, 1]


def test_bake_eye_patches_reports_centre_that_selects_no_faces(eye_triangles):
    import numpy as np

    positions, coords, centres = eye_triangles
    centres[1, 0] = 10
    source = np.full((2, 2, 3), .05, dtype=np.float32)
    with pytest.raises(ValueError, match="eye 1 selected no faces"):
        paint_eyes.bake_eye_patches(
            positions, coords, centres, source, source, 1, 16, 0, paint_eyes.EyeStyle()
        )
