"""Stand in for Blender's Finish command while testing Studio's worker lifecycle."""

import shutil
import sys
import time
from pathlib import Path

source, result, ready, release = map(Path, sys.argv[1:5])
write_before_release = sys.argv[5] == "complete"

if write_before_release:
    shutil.copyfile(source, result)
print("I2L_STAGE::retopologise::Retopologising to 40,000 faces", flush=True)
ready.touch()

deadline = time.monotonic() + 30
while not release.is_file():
    if time.monotonic() >= deadline:
        raise SystemExit("the test never released the Finish worker")
    time.sleep(0.02)

print("I2L_STAGE::done::Finished at 0.0 MB", flush=True)
