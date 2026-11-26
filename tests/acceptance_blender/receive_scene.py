"""Receive and execute one shipped live-worker request in a fake Blender scene."""

import contextlib
import io
import json
import socket
import sys
from types import ModuleType, SimpleNamespace


def scene_object(name, position):
    return SimpleNamespace(name=name, matrix_world=SimpleNamespace(translation=position))


def read_request(connection):
    request = bytearray()
    while chunk := connection.recv(4096):
        request.extend(chunk)
        try:
            return json.loads(request)
        except json.JSONDecodeError:
            continue
    raise ValueError("Client closed before sending a complete request")


def execute(code, mode):
    bpy = ModuleType("bpy")
    objects = [
        scene_object("Camera", (9, 9, 9)),
        scene_object("JOINT_head", (1.234567, -2.5, 0.125)),
        scene_object("JOINT_patère", (-3, 0.000007, 4)),
    ]
    if mode == "error":
        objects.append(SimpleNamespace(name="JOINT_unreadable"))
    bpy.data = SimpleNamespace(objects=objects)
    sys.modules["bpy"] = bpy

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(code, "<received worker>", "exec"), {})
    except Exception as error:
        if mode != "error":
            raise
        return {"status": "error", "message": str(error)}

    if mode == "error":
        raise AssertionError("The worker unexpectedly read the broken marker")
    # Exceed the sender's recv buffer so a single read cannot contain the whole reply.
    return {"status": "success", "result": {"result": output.getvalue()}, "padding": " " * 65536}


def main():
    descriptor, mode = sys.argv[1:]
    with socket.socket(fileno=int(descriptor)) as listener:
        listener.settimeout(10)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(10)
            request = read_request(connection)
            assert request["type"] == "execute_code"
            reply = json.dumps(execute(request["params"]["code"], mode)).encode()
            if mode == "truncated":
                reply = reply[:-1]
            middle = len(reply) // 2
            connection.sendall(reply[:middle])
            connection.sendall(reply[middle:])


if __name__ == "__main__":
    main()
