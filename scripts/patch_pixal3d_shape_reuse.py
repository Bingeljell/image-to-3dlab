#!/usr/bin/env python3
"""Let Pixal3D save a run's shape and later repaint it from new pictures, via `PIXAL3D_SAVE_SHAPE` / `PIXAL3D_LOAD_SHAPE`.

    python scripts/patch_pixal3d_shape_reuse.py     # then rebuild trellis-cli
    PIXAL3D_SAVE_SHAPE=run.shape.bin trellis-cli ... run.glb          # build, keep the shape
    PIXAL3D_LOAD_SHAPE=run.shape.bin trellis-cli ... recoloured.glb   # paint only

**Why.** Pixal3D builds a shape first (structure flow, then shape flow) and paints it
last (texture flow). The painter only needs the finished shape latent and its voxel
coordinates, which fit in about 650 KB. Saving them lets a later run skip straight to
painting from a different picture of the same pose: the mesh, its UVs after Finish and the
normal map all come out identical, so a rig made for one paint fits every other. A
repaint takes about a third of a full run.

**The file.** int N, int res, N x int[3] high-res voxel coordinates, then N x 32 floats of
the NORMALISED shape latent (what the texture flow reads). Read back, the run skips stages
2 and 3 and carries on from the shape decode exactly as a normal run would. Only the
multi-view route (`--views`, and `--sv-image`, which wraps it) is patched; that is the
only Pixal3D route this repo runs.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone, then
`cmake --build vendor/pixal3d-cpp/build --target trellis-cli`. Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "i2l_shape_reuse"
SAVE_ENV = "PIXAL3D_SAVE_SHAPE"
LOAD_ENV = "PIXAL3D_LOAD_SHAPE"

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "vendor" / "pixal3d-cpp" / "src" / "trellis_cli.cpp"

INCLUDE_ANCHOR = "#include <filesystem>\n"
# What the injected code calls. Never assume the host file's includes.
INCLUDE_ADDITION = f"#include <cstdlib>   // {MARKER}: std::getenv\n"

LOAD_BLOCK = f'''    // {MARKER}: {LOAD_ENV}=<file> skips stages 2-3 and paints a saved shape
    // (written by {SAVE_ENV}=<file>) from this run's pictures.
    // File: int N, int res, N x int[3] HR coords, N x 32 float normalised HR shape SLAT.
    const char* i2l_load_path = std::getenv("{LOAD_ENV}");
    const bool i2l_reuse = i2l_load_path && *i2l_load_path;
    int i2l_res = 0; vector<std::array<int,3>> i2l_shc; vector<float> i2l_norm;
    if (i2l_reuse) {{
        FILE* f = fopen(i2l_load_path, "rb");
        int n = 0;
        if (!f || fread(&n,4,1,f) != 1 || fread(&i2l_res,4,1,f) != 1 || n <= 0) {{
            fprintf(stderr, "[trellis] cannot read saved shape %s\\n", i2l_load_path); if (f) fclose(f); return 1; }}
        i2l_shc.resize(n); i2l_norm.resize((size_t)32 * n);
        for (auto& c : i2l_shc) {{ int xyz[3]; if (fread(xyz,4,3,f) != 3) {{ fclose(f); return 1; }} c = {{xyz[0],xyz[1],xyz[2]}}; }}
        const size_t got = fread(i2l_norm.data(), 4, i2l_norm.size(), f); fclose(f);
        if (got != i2l_norm.size()) {{ fprintf(stderr, "[trellis] saved shape %s is short\\n", i2l_load_path); return 1; }}
        printf("[2/6] [3/6] reusing saved shape %s (N=%d @res%d): repaint only\\n", i2l_load_path, n, i2l_res);
    }}

'''

SAVE_BLOCK = f'''    if (const char* sp = std::getenv("{SAVE_ENV}"); sp && *sp && !i2l_reuse) {{   // {MARKER}
        FILE* f = fopen(sp, "wb");
        if (!f) {{ fprintf(stderr, "[trellis] cannot write saved shape %s\\n", sp); return 1; }}
        int n = N, res = RES; fwrite(&n,4,1,f); fwrite(&res,4,1,f);
        for (auto& c : shc) {{ int xyz[3] = {{c[0],c[1],c[2]}}; fwrite(xyz,4,3,f); }}
        fwrite(slat_norm.data(),4,slat_norm.size(),f); fclose(f);
        printf("      [shape] saved %s: N=%d res=%d\\n", sp, n, res);
    }}
'''

# (name, upstream text, replacement). Each anchor must appear exactly once.
EDITS = [
    ("stage 2",
     ('    printf("[2/6] SS proj conditioning + flow\\n");\n'
     "    vector<std::array<int,3>> coords;\n"
     "    {\n"),
     (LOAD_BLOCK
     + '    if (!i2l_reuse) printf("[2/6] SS proj conditioning + flow\\n");\n'
     "    vector<std::array<int,3>> coords;\n"
     "    if (!i2l_reuse) {\n")),
    # The single-image route has the same check after a shorter printf; the timing pins this one.
    ("empty voxels",
     ('    printf("      active voxels @res32 = %d  (%.1fs)\\n", (int)coords.size(), now() - t_stage);\n'
     '    if (coords.empty()) { fprintf(stderr, "no voxels produced\\n"); return 1; }\n'),
     ('    printf("      active voxels @res32 = %d  (%.1fs)\\n", (int)coords.size(), now() - t_stage);\n'
     '    if (!i2l_reuse && coords.empty()) { fprintf(stderr, "no voxels produced\\n"); return 1; }\n')),
    ("stage 3",
     ('    printf("[3/6] shape SLAT flow (LR 512 -> upsample -> HR %d cascade, max_tok=%d)\\n", '
     "cfg.hr_res, cfg.max_tokens);\n"
     "    vector<float> lr_norm, lr_dn;\n"
     "    {\n"),
     ('    if (!i2l_reuse) printf("[3/6] shape SLAT flow (LR 512 -> upsample -> HR %d cascade, '
     'max_tok=%d)\\n", cfg.hr_res, cfg.max_tokens);\n'
     "    vector<float> lr_norm, lr_dn;\n"
     "    if (!i2l_reuse) {\n")),
    ("upsample",
     ("    vector<std::array<int,3>> hr_coords;\n"
     '    { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);\n'),
     ("    vector<std::array<int,3>> hr_coords;\n"
     '    if (!i2l_reuse) { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);\n')),
    ("grid",
     ("        hr_res -= 128;\n"
     "    }\n"
     "    const int grid = hr_res / 16;\n"),
     ("        hr_res -= 128;\n"
     "    }\n"
     "    if (i2l_reuse) { hr_res = i2l_res; shc = i2l_shc; }\n"
     "    const int grid = hr_res / 16;\n")),
    ("hr shape flow",
     ("    vector<float> slat_norm;\n"
     "    {\n"),
     ("    vector<float> slat_norm;\n"
     "    if (i2l_reuse) slat_norm = i2l_norm;\n"
     "    else {\n")),
    ("save",
     ('    cond_lap("denorm + stats (+ dump)");\n'
     '    cond_stage_end("HR shape SLAT");\n'),
     (SAVE_BLOCK
     + '    cond_lap("denorm + stats (+ dump)");\n'
     '    cond_stage_end("HR shape SLAT");\n')),
]


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent; refuses if any anchor is missing or doubled."""
    if is_patched(source):
        return source
    for name, anchor, _ in EDITS:
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(
                f"{name} anchor found {count} times in trellis_cli.cpp (expected 1): the "
                "multi-view route changed upstream. Check whether it grew its own way to "
                "reuse a shape before patching."
            )
    if INCLUDE_ADDITION not in source:
        if source.count(INCLUDE_ANCHOR) != 1:
            raise SystemExit("include anchor not found exactly once in trellis_cli.cpp")
        source = source.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + INCLUDE_ADDITION, 1)
    for _, anchor, replacement in EDITS:
        source = source.replace(anchor, replacement, 1)
    return source


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
    print(f"patched: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
