#!/usr/bin/env python3
"""Teach the vendored TRELLIS.2 to condition on several views of one subject.

Single-view generation has to invent everything it cannot see, which is where our worst
defects come from: a beer mug given three handles, a back of a head that is mostly
absent, soft faces. More views replace guessing with evidence.

The plumbing is almost all there already. `run()` calls `get_cond([image], 512)`, and
`image_feature_extractor.__call__` accepts a list of PIL images and returns `(B, N, D)`.
Only two things are missing:

1. `run()` takes a single image, so a list never reaches the conditioning path.
2. With B views the encoder returns a *batch* of B feature sets, while
   `sample_sparse_structure` builds noise of shape `(num_samples, ...)` and expects one
   conditioning. Left as a batch, the model would be asked for B separate objects. The
   views must be concatenated along the **token** axis into `(1, B*N, D)` -- one longer
   sequence meaning "here is everything known about this object".

Views must be the same pose from an orbiting camera. Different poses (sitting vs
standing) describe different objects and break reconstruction.

The vendored checkout is git-ignored, so this is a re-runnable patch script rather than
an edit in place, matching `patch_trellis_no_bria.py`. Safe to run twice.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRELLIS = ROOT / "vendor" / "trellis-mac"
PIPELINE = TRELLIS / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
GENERATE = TRELLIS / "generate.py"


def replace(path: Path, needle: str, replacement: str, marker: str) -> bool:
    """Apply one substitution, skipping if `marker` shows it is already applied."""
    source = path.read_text()
    if marker in source:
        return False
    if needle not in source:
        raise RuntimeError(f"expected anchor not found in {path.name}:\n{needle[:120]}")
    path.write_text(source.replace(needle, replacement, 1))
    return True


def patch_get_cond() -> bool:
    needle = """        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()"""
    replacement = """        cond = self.image_cond_model(image)
        # Several views describe ONE object, so fold them into a single conditioning
        # sequence along the token axis: (B, N, D) -> (1, B*N, D). Leaving them batched
        # would ask the sampler for B separate objects, since it builds noise shaped
        # (num_samples, ...) and pairs it with the conditioning batch.
        if cond.ndim == 3 and cond.shape[0] > 1:
            cond = cond.reshape(1, -1, cond.shape[-1])
        if self.low_vram:
            self.image_cond_model.cpu()"""
    return replace(PIPELINE, needle, replacement, "Several views describe ONE object")


def patch_run() -> bool:
    needle = """        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None"""
    replacement = """        # Accept either one image or several views of the same subject.
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if preprocess_image:
            images = [self.preprocess_image(i) for i in images]
        torch.manual_seed(seed)
        cond_512 = self.get_cond(images, 512)
        cond_1024 = self.get_cond(images, 1024) if pipeline_type != '512' else None"""
    return replace(PIPELINE, needle, replacement, "Accept either one image or several")


def patch_generate_arg() -> bool:
    needle = '    parser.add_argument("image", help="Path to input image")'
    replacement = (
        '    parser.add_argument(\n'
        '        "image", nargs="+",\n'
        '        help="Input image, or several views of the same subject in the same "\n'
        '             "pose from an orbiting camera",\n'
        '    )'
    )
    return replace(GENERATE, needle, replacement, 'nargs="+"')


def patch_generate_load() -> bool:
    needle = """    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")"""
    replacement = """    img = [PILImage.open(path) for path in args.image]
    for path, view in zip(args.image, img):
        print(f"Input: {path} ({view.size[0]}x{view.size[1]})")
    if len(img) == 1:
        img = img[0]
    else:
        print(f"Conditioning on {len(img)} views")"""
    return replace(GENERATE, needle, replacement, "Conditioning on")


def patch_generate_exists() -> bool:
    needle = """    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")"""
    replacement = """    missing = [p for p in args.image if not os.path.exists(p)]
    if missing:
        print(f"Error: {', '.join(missing)} not found")"""
    return replace(GENERATE, needle, replacement, "missing = [p for p in args.image")


def main() -> int:
    if not PIPELINE.exists() or not GENERATE.exists():
        raise SystemExit(
            "vendored TRELLIS.2 not found; bootstrap it before applying this patch"
        )
    steps = (
        ("get_cond token concatenation", patch_get_cond),
        ("run() accepts a view list", patch_run),
        ("generate.py accepts several paths", patch_generate_arg),
        ("generate.py loads every view", patch_generate_load),
        ("generate.py existence check", patch_generate_exists),
    )
    for label, step in steps:
        print(f"  {'applied ' if step() else 'already '}: {label}")
    print("multi-view patch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
