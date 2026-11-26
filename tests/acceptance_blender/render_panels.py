"""Stand in for the renderer and produce small PNGs with distinct foregrounds."""

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = ((232, 65, 51), (59, 192, 88), (53, 115, 242), (215, 155, 45))


def main():
    args = sys.argv[1:]
    source, output, _, _, _, _, shots = args[args.index("--") + 1 :]
    source = Path(source)
    output = Path(output)
    assert source.read_bytes()[:4] == b"glTF"
    assert output.is_dir()

    with Path(os.environ["AF_TEST_RENDER_LOG"]).open("a", encoding="utf-8") as log:
        log.write(json.dumps(args) + "\n")
    mode = os.environ["AF_TEST_RENDER_MODE"]
    for index, shot in enumerate(shots.split(";")):
        name, _, _ = shot.split("=")
        if mode == "missing" and index == 3:
            continue
        panel = Image.new("RGB", (100, 100), (12, 12, 12))
        ImageDraw.Draw(panel).rectangle((30, 30, 69, 69), fill=COLORS[index])
        panel.save(output / f"{name}.png")
    if mode == "nonzero":
        print("fixture renderer exited after writing the panels", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
