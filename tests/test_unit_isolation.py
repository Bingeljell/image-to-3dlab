"""Exercise the guards through real socket and process calls."""

import os
import shutil
import socket
import subprocess
import sys

import pytest


def _windows_root():
    return {"SYSTEMROOT": os.environ["SYSTEMROOT"]} if sys.platform == "win32" else {}


@pytest.mark.parametrize("action", [
    lambda: socket.getaddrinfo("external.invalid", 443),
    lambda: socket.create_connection(("192.0.2.1", 443)),
    lambda: subprocess.run(["blender", "--background"], check=True),
    lambda: subprocess.run(["curl", "https://external.invalid/weights"], check=True),
    lambda: subprocess.run(["bash", "-c", "printf unexpected"], check=True),
    lambda: subprocess.run("printf unexpected", shell=True, check=True),
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
