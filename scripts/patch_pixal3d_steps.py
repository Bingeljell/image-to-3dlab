#!/usr/bin/env python3
"""Let Pixal3D (pixal3d.cpp) run fewer sampling steps, via `PIXAL3D_STEPS=N`.

    python scripts/patch_pixal3d_steps.py            # then rebuild trellis-cli
    python scripts/pixal3d_generate.py in.png out.glb --steps 8

**Why.** Every flow stage in `trellis-cli` hard-codes `sp.steps = 12`, with no flag to
change it. On a busy hard-surface image the HR shape and texture flows run ~16k tokens,
and a base Apple GPU spends ~40 s per forward on them: 12 steps is most of a 27-minute
run. The sampler itself builds its timestep schedule from `sp.steps`
(`src/flow_runner.cpp`), and TRELLIS.2's own demo exposes steps as a slider, so fewer
steps is a supported setting rather than a hack. Whether 8 looks good enough is a
per-asset judgement, which is why this is opt-in.

The override lives in `sample_flow`, the one function every stage samples through, so a
single change covers the SS, shape and texture flows. Unset, or anything that is not a
whole number from 1 to 50, leaves each stage's own count alone. When it applies, the log
says so on every flow, so a run log always records what it actually did.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone, then
`cmake --build vendor/pixal3d-cpp/build --target trellis-cli`. Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "i2l_steps"
ENV_VAR = "PIXAL3D_STEPS"
MAX_STEPS = 50

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "vendor" / "pixal3d-cpp" / "src" / "flow_runner.cpp"

SIGNATURE_ANCHOR = (
    "std::vector<float> sample_flow(const FlowFwdProj& fwd, std::vector<float> sample,\n"
)
PARAM_ANCHOR = (
    "                               const SamplerParams& sp,\n"
    "                               std::vector<std::vector<float>>* trace) {\n"
    "    const float sm = sp.sigma_min;\n"
)
PARAM_REPLACEMENT = (
    "                               const SamplerParams& sp_in,\n"
    "                               std::vector<std::vector<float>>* trace) {\n"
    f"    // {MARKER}: {ENV_VAR}=N overrides the stage's hard-coded step count.\n"
    "    SamplerParams sp = sp_in;\n"
    f"    sp.steps = {MARKER}_override(sp_in.steps, std::getenv(\"{ENV_VAR}\"));\n"
    "    if (sp.steps != sp_in.steps)\n"
    f"        printf(\"      [flow] {ENV_VAR}=%d overrides %d steps\\n\", sp.steps, sp_in.steps);\n"
    "    const float sm = sp.sigma_min;\n"
)
HELPER = (
    f"// {MARKER}: parse {ENV_VAR}. Anything but a whole number 1..{MAX_STEPS} keeps the default.\n"
    f"static int {MARKER}_override(int fallback, const char* raw) {{\n"
    "    if (!raw || !*raw) return fallback;\n"
    "    char* end = nullptr;\n"
    "    const long v = std::strtol(raw, &end, 10);\n"
    f"    if (*end != '\\0' || v < 1 || v > {MAX_STEPS}) return fallback;\n"
    "    return (int)v;\n"
    "}\n\n"
)
# Everything the injected code calls. Never assume the host file's includes.
REQUIRED_INCLUDES = ("#include <cstdlib>", "#include <cstdio>")


def steps_override(fallback: int, raw: str | None) -> int:
    """The same decision the injected C++ makes, for tests and for the wrapper.

    `PIXAL3D_STEPS=0`, `=abc` or `=8.5` must not silently change the run.
    """
    # Mirrors strtol + `*end == '\0'`: leading whitespace and a sign are allowed, anything
    # after the digits is not. Python's int() is laxer (trailing spaces, `1_0`).
    if not raw or raw != raw.rstrip() or "_" in raw:
        return fallback
    try:
        value = int(raw.lstrip(), 10)
    except ValueError:
        return fallback
    return value if 1 <= value <= MAX_STEPS else fallback


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    if is_patched(source):
        return source
    for anchor in (SIGNATURE_ANCHOR, PARAM_ANCHOR):
        if anchor not in source:
            raise SystemExit(
                "anchor not found in flow_runner.cpp: sample_flow's signature changed "
                "upstream. Check whether it grew its own steps option before patching."
            )
    missing = [inc for inc in REQUIRED_INCLUDES if inc not in source]
    if missing:
        raise SystemExit(f"flow_runner.cpp lacks {missing}; the injected code needs them.")
    source = source.replace(SIGNATURE_ANCHOR, HELPER + SIGNATURE_ANCHOR, 1)
    return source.replace(PARAM_ANCHOR, PARAM_REPLACEMENT, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text(encoding="utf-8")

    if args.check:
        print(("patched: " if is_patched(source) else "NOT PATCHED: ") + str(args.target))
        return 0 if is_patched(source) else 1

    patched = apply(source)
    if patched == source:
        print(f"already patched: {args.target}")
        return 0
    args.target.write_text(patched, encoding="utf-8")
    print(f"patched {args.target}: {ENV_VAR}=N now sets the step count. Rebuild trellis-cli.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
