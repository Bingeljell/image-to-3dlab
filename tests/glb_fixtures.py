"""A minimal but valid GLB (a JSON chunk and a BIN chunk), for tests that rewrite one.

Real files come from Blender or a backend; the code that sets `asset.generator` must still
see a well-formed file, so these fixtures are parsed the same way, not faked bytes.
"""

from __future__ import annotations

import json
import struct

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def valid_glb(*, generator: str = "test exporter", asset_extra: dict | None = None,
              binary: bytes = b"\x00\x00\x00\x00") -> bytes:
    asset = {"version": "2.0", "generator": generator}
    asset.update(asset_extra or {})
    payload = json.dumps({"asset": asset}).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    json_chunk = struct.pack("<II", len(payload), JSON_CHUNK) + payload
    bin_chunk = struct.pack("<II", len(binary), BIN_CHUNK) + binary
    body = json_chunk + bin_chunk
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(body)) + body


def read_glb(data: bytes) -> tuple[dict, bytes]:
    """(JSON document, BIN chunk) from a GLB."""
    magic, version, _ = struct.unpack_from("<III", data, 0)
    assert (magic, version) == (GLB_MAGIC, 2)
    offset, document, binary = 12, None, b""
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8:offset + 8 + length]
        if kind == JSON_CHUNK:
            document = json.loads(chunk)
        elif kind == BIN_CHUNK:
            binary = chunk
        offset += 8 + length
    return document, binary
