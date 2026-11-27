"""Exercise the guards through real socket and process calls."""

import os
import platform
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import unit_isolation


def _windows_root():
    return {"SYSTEMROOT": os.environ["SYSTEMROOT"]} if sys.platform == "win32" else {}


@pytest.mark.parametrize("action", [
    lambda: socket.getaddrinfo("external.invalid", 443),
    lambda: socket.create_connection(("192.0.2.1", 443)),
    lambda: subprocess.run(["blender", "--background"], check=True),
    lambda: subprocess.run(["curl", "https://external.invalid/weights"], check=True),
    lambda: subprocess.run(["bash", "-c", "printf unexpected"], check=True),
    lambda: subprocess.run("printf unexpected", shell=True, check=True),
    lambda: subprocess.run(["/sbin/ldconfig", "-p"], shell=True, check=True),
    lambda: subprocess.run("ver", shell=True, check=True),
    lambda: subprocess.run(["fc-list", "--help"], check=True),
    lambda: subprocess.run(["/usr/sbin/system_profiler", "SPFontsDataType"], check=True),
    lambda: subprocess.run([sys.executable, "download_weights.py"], check=True),
    lambda: subprocess.run([sys.executable, "scripts/bootstrap_pixal3d.py"], check=True),
    lambda: subprocess.run([sys.executable, "scripts/pixal3d_generate.py"], check=True),
    lambda: socket.create_connection(("127.0.0.1", 9876)),
])
def test_test_process_refuses_external_work(action):
    with pytest.raises(RuntimeError, match="Offline tests"):
        action()


def test_datagram_socket_refuses_external_destination():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        with pytest.raises(RuntimeError, match="Offline tests"):
            client.sendto(b"blocked", ("192.0.2.1", 443))


def test_loopback_socket_still_works():
    with socket.socket() as server, socket.socket() as client:
        server.bind(("127.0.0.1", 0))
        server.listen()
        client.connect(server.getsockname())
        with server.accept()[0] as connection:
            connection.sendall(b"ok")
            assert client.recv(2) == b"ok"


def test_python_children_retain_guards_with_replacement_environment(tmp_path):
    probe = """
import socket
import subprocess
import sys

for action in (lambda: socket.getaddrinfo('external.invalid', 443),
               lambda: subprocess.run(['blender', '--background'])):
    try:
        action()
    except RuntimeError as exc:
        assert 'Offline tests' in str(exc)
    else:
        raise AssertionError('external action was allowed')
"""
    parent = """
import subprocess
import sys

subprocess.run([sys.executable, '-c', sys.argv[1]], check=True)
"""
    subprocess.run([sys.executable, "-c", parent, probe], cwd=tmp_path,
                   env={"PYTHONPATH": str(tmp_path), **_windows_root()}, check=True, timeout=15)


def test_node_fixture_child_refuses_external_work(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node")
    program = """
import assert from 'node:assert/strict';
import { lookup, promises } from 'node:dns';
import { connect, createServer } from 'node:net';
import { spawn } from 'node:child_process';
assert.throws(() => lookup('external.invalid', () => {}), /Offline tests/);
assert.throws(() => connect(443, '192.0.2.1'), /Offline tests/);
assert.throws(() => spawn('blender', ['--background']), /Offline tests/);
await assert.rejects(promises.lookup('external.invalid'), /Offline tests/);
const server = createServer(socket => socket.end('ok'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const client = connect(server.address().port, '127.0.0.1');
assert.equal(await new Promise(resolve => client.once('data', resolve)).then(String), 'ok');
server.close();
"""
    subprocess.run([node, "--input-type=module", "--eval", program], cwd=tmp_path,
                   env={"NODE_OPTIONS": "", **_windows_root()}, check=True, timeout=15)


@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_only_linux_allows_the_exact_glibc_cache_query(monkeypatch, system):
    monkeypatch.setattr(unit_isolation, "sys", SimpleNamespace(platform=system))
    command = ["/sbin/ldconfig", "-p"]
    if system == "linux":
        unit_isolation._audit("subprocess.Popen", (command[0], command, None, None))
    else:
        with pytest.raises(RuntimeError, match="Offline tests block process"):
            unit_isolation._audit("subprocess.Popen", (command[0], command, None, None))
    for executable, args in [
        ("/sbin/ldconfig", ["/sbin/ldconfig"]),
        ("/sbin/ldconfig", [*command, "-i"]),
        ("/sbin/ldconfig", ["git", "-p"]),
        ("ldconfig", command),
        ("/usr/sbin/ldconfig", command),
        ("git", command),
        ("/tmp/ldconfig", command),
    ]:
        with pytest.raises(RuntimeError, match="Offline tests block process"):
            unit_isolation._audit("subprocess.Popen", (executable, args, None, None))


@pytest.mark.skipif(sys.platform != "linux", reason="glibc library cache on Linux")
def test_linux_child_reads_glibc_cache_before_torch_import():
    subprocess.run(
        [sys.executable, "-c", """
import ctypes.util
import subprocess

subprocess.run(['/sbin/ldconfig', '-p'], check=True, stdout=subprocess.DEVNULL)
ctypes.util.find_library('dl')
import torch
assert torch.ones(1).item() == 1
"""],
        check=True, timeout=30,
    )


@pytest.mark.parametrize("system", ["win32", "darwin"])
def test_native_windows_version_fallback_is_platform_specific(monkeypatch, system):
    def unexpected_shell(*args, **kwargs):
        pytest.fail("legacy Windows version probe reached")

    monkeypatch.setattr(unit_isolation, "sys", SimpleNamespace(platform=system))
    monkeypatch.setattr(platform, "_syscmd_ver", unexpected_shell, raising=False)
    unit_isolation._use_native_windows_version()
    if system == "win32":
        assert platform._syscmd_ver("foo", version="bar") == ("foo", "", "bar")
    else:
        assert platform._syscmd_ver is unexpected_shell


def test_windows_optional_lscpu_probe_is_unavailable_without_starting_a_process(monkeypatch):
    monkeypatch.setattr(unit_isolation, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(FileNotFoundError, match="lscpu"):
        unit_isolation._audit("subprocess.Popen", (None, "lscpu", None, None))


@pytest.mark.parametrize(("executable", "command"), [
    (None, "lscpu --help"),
    (None, r"C:\fake\lscpu.exe"),
    ("git.exe", "lscpu"),
    ("lscpu.exe", "lscpu"),
    (None, '"git.exe"malformed'),
    (None, "cmd.exe /c ver"),
    (None, "blender --background"),
    (None, "curl.exe https://external.invalid/weights"),
    (None, r'python.exe "C:\test data\download_weights.py"'),
])
def test_windows_command_line_still_obeys_process_policy(monkeypatch, executable, command):
    monkeypatch.setattr(unit_isolation, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(RuntimeError, match="Offline tests"):
        unit_isolation._audit("subprocess.Popen", (executable, command, None, None))


@pytest.mark.parametrize("command", [
    "git --version",
    r'"C:\Program Files\Git\cmd\git.exe" --version',
    r'"C:\Python\python.exe" -c "print(1)"',
])
def test_windows_command_line_recovers_ordinary_fixture_tools(monkeypatch, command):
    monkeypatch.setattr(unit_isolation, "sys", SimpleNamespace(platform="win32"))
    unit_isolation._audit("subprocess.Popen", (None, command, None, None))


def test_windows_rejects_unstructured_arguments_before_starting_process(monkeypatch):
    monkeypatch.setattr(unit_isolation, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(RuntimeError, match="Offline tests block raw Windows command line"):
        subprocess.run('python.exe download_"weights.py"', check=True)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows version lookup")
def test_windows_child_checks_native_version_and_imports_dependencies():
    subprocess.run(
        [sys.executable, "-c", """
import platform
import subprocess
import sys

native = sys.getwindowsversion()
release, version, _, _ = platform.win32_ver()
assert release
assert version == '.'.join(map(str, native.platform_version or native[:3]))
assert platform.system() == 'Windows'
assert platform.machine()
try:
    subprocess.run('lscpu', capture_output=True, text=True)
except FileNotFoundError:
    pass
else:
    raise AssertionError('Windows should treat lscpu as unavailable')
import numpy.testing
import scipy.linalg
import trimesh
import torch
assert torch.ones(1).item() == 1
"""],
        check=True, timeout=30,
    )
