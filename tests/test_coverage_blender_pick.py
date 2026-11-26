"""Check the pixel locations passed to Blender's camera unprojection."""

import blender_pick_pixel as pick
import pytest


@pytest.mark.parametrize(
    ("pixel", "expected"),
    [
        ((0, 0), (-0.75, 0.5)),
        ((3, 1), (0.75, -0.5)),
        ((1.5, 0.5), (0, 0)),
    ],
)
def test_pixel_centres_map_from_top_left_into_camera_space(pixel, expected):
    assert pick.pixel_ndc(*pixel, 4, 2) == expected


@pytest.mark.parametrize("complete", [True, False])
def test_pick_cli_returns_fragmented_mcp_response(monkeypatch, capsys, complete):
    import json
    import sys

    class Connection:
        def __init__(self):
            self.fragments = iter([b'{"status":', b'"success"}' if complete else b""])
            self.timeout = None
            self.request = None
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def settimeout(self, value):
            self.timeout = value

        def sendall(self, data):
            self.request = json.loads(data)

        def recv(self, size):
            assert size == 65536
            return next(self.fragments)

    connection = Connection()

    def connect(address, *, timeout):
        assert address == ("localhost", 4444)
        assert timeout == 10
        return connection

    monkeypatch.setattr(pick.socket, "create_connection", connect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pick",
            "0",
            "0",
            "--width",
            "4",
            "--height",
            "2",
            "--host",
            "localhost",
            "--port",
            "4444",
        ],
    )
    import runpy

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(pick.__file__, run_name="__main__")
    assert stopped.value.code == 0
    response = capsys.readouterr().out.strip()
    if complete:
        assert json.loads(response) == {"status": "success"}
    else:
        assert response == '{"status":'
    assert connection.closed and connection.timeout == 60
    assert connection.request["type"] == "execute_code"
    compile(connection.request["params"]["code"], "<pick request>", "exec")


def test_packaged_pick_rejects_unknown_camera_view():
    from scripts import blender_pick_pixel

    with pytest.raises(KeyError, match="unknown"):
        blender_pick_pixel.build_code(0, 0, 4, 2, "unknown")
