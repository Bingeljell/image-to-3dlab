"""Check a Blender frame sequence and encode it to MP4.

Run (internal): imported by live Blender video scripts.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def encode_frames(frame_dir: Path, label: str, frames: int, fps: int, output: Path) -> None:
    expected = {f"{label}_{index:04d}.png" for index in range(frames)}
    actual = {
        path.name for path in frame_dir.iterdir()
        if path.is_file() and path.name.startswith(f"{label}_") and path.suffix == ".png"
    }
    if actual != expected:
        details = []
        if missing := expected - actual:
            details.append(f"missing {len(missing)} (including {min(missing)})")
        if extra := actual - expected:
            details.append(f"found {len(extra)} unexpected (including {min(extra)})")
        raise ValueError(f"expected {frames} consecutive frames starting at 0000, " + ", ".join(details))

    pattern = str(frame_dir / f"{label}_").replace("%", "%%") + "%04d.png"
    with tempfile.TemporaryDirectory(prefix=".video_", dir=output.parent) as staging:
        staged_output = Path(staging) / output.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-xerror",
                    "-framerate", str(fps),
                    "-start_number", "0",
                    "-start_number_range", "1",
                    "-i", pattern,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    # x264 needs even dimensions.
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-crf", "18",
                    "-movflags", "+faststart",
                    str(staged_output),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg could not encode the video (exit {exc.returncode})") from exc
        staged_output.replace(output)
