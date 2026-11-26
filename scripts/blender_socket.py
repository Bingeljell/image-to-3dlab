"""Send a command and check the BlenderMCP reply.

Run (internal): imported by live Blender scripts.
"""

from __future__ import annotations

import json
import socket


def execute_code(code: str, host: str, port: int, timeout: float) -> str:
    """Return the original JSON only after Blender confirms success."""
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(timeout)
        connection.sendall(json.dumps(request).encode("utf-8"))
        data = bytearray()
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError as exc:
                raise RuntimeError("Blender timed out before sending a complete reply") from exc
            if not chunk:
                raise RuntimeError("Blender closed before sending a complete reply")
            data.extend(chunk)
            try:
                reply = data.decode("utf-8")
                envelope = json.loads(reply)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            break

    if isinstance(envelope, dict):
        if envelope.get("status") == "error":
            raise RuntimeError(envelope.get("message") or "Blender reported an error")
        if envelope.get("status") == "success" and isinstance(result := envelope.get("result"), dict):
            if result.get("executed") is True:
                return reply
            if "error" in result:
                raise RuntimeError(result["error"] or "Blender failed to execute the command")
    raise RuntimeError("unexpected Blender reply")
