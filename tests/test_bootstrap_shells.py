"""The old Mac shell installers hand off to the Python ones that ask first."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHELLS = {
    "bootstrap_macos.sh": "bootstrap_sf3d.py",
    "bootstrap_trellis_macos.sh": "bootstrap_trellis_space_macos.py",
}


def test_the_shell_installers_name_the_python_ones_and_do_not_clone():
    sf3d = (REPO / "scripts" / "bootstrap_macos.sh").read_text()
    trellis = (REPO / "scripts" / "bootstrap_trellis_macos.sh").read_text()
    assert "exec python3" in sf3d and "bootstrap_sf3d.py" in sf3d and "exit 1" in sf3d
    assert "exec python3" in trellis and "bootstrap_trellis_space_macos.py" in trellis
    assert "exit 1" in trellis
    for text in (sf3d, trellis):
        assert "git clone" not in text
        assert "pip " not in text and "brew " not in text


def _fake_python(bin_dir: Path, record: Path) -> None:
    bin_dir.mkdir()
    program = bin_dir / "python3"
    program.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(record))}\n"
        "exit 7\n"
    )
    program.chmod(0o755)


def test_the_shell_runs_the_python_installer_and_forwards_args(tmp_path):
    bindir = tmp_path / "bin"
    record = tmp_path / "args"
    _fake_python(bindir, record)
    env = os.environ.copy()
    env["PATH"] = f"{bindir}:/usr/bin:/bin"
    for name, installer in SHELLS.items():
        record.unlink(missing_ok=True)
        result = subprocess.run(
            ["/bin/bash", str(REPO / "scripts" / name), "--flag", "value"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 7, result.stderr
        lines = record.read_text().splitlines()
        assert lines[0].endswith(f"scripts/{installer}")
        assert lines[1:] == ["--flag", "value"]


def test_a_missing_python_exits_and_names_the_installer(tmp_path):
    # /usr/bin/python3 exists on macOS, so the path cannot include it.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("dirname", "pwd"):
        source = next(
            path for path in (Path("/bin") / name, Path("/usr/bin") / name) if path.exists()
        )
        (bindir / name).symlink_to(source)
    env = os.environ.copy()
    env["PATH"] = str(bindir)
    for name, installer in SHELLS.items():
        result = subprocess.run(
            ["/bin/bash", str(REPO / "scripts" / name)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert installer in result.stderr
        assert result.stdout == ""
