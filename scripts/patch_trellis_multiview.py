#!/usr/bin/env python3
"""Teach the vendored TRELLIS.2 to condition on several views of one subject.

TRELLIS.2 ships single-image only: there is no multi-image entry point, and no camera
or view embeddings anywhere in the conditioning path. Microsoft's own tracker carries
"multi-image inputs is worse" (issue #103) with no root cause and no fix.

**A first attempt concatenated the per-view feature tokens into one long sequence.
That is the wrong technique and it measurably damages the result.** Without view
embeddings the model cannot tell which token came from which view, so it reconciles
them as a single observation. Front and back are the maximally contradictory pair -- a
face and a back-of-skull share almost no content -- so averaging wrecks the face. We
reproduced exactly that: the back of the head improved markedly, the front fell apart.

TRELLIS v1 solved this properly and tuning-free, and v2 dropped it. Neither of its
modes touches the conditioning tokens; both leave the views separate and combine the
denoiser's *predictions*:

  multidiffusion  run the denoiser once per view each step, average the predictions
  stochastic      use one view per step, cycling through them

This restores both. Note Microsoft's caveat on v1: it is "a tuning-free algorithm
without training a specialized model, so it may not give the best results for all
input images". Multi-view here is an inference-time trick, never a trained capability.

Views must be the same pose from an orbiting camera. Different poses describe different
objects. Opposed views (front/back) maximise coverage; overlapping views (two
three-quarters) contradict each other less.

Nothing here needs CUDA -- it is sampler-level tensor work, unlike the `cumesh` and
`flex_gemm` mesh ops our Mac port has to stub out.

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
        raise RuntimeError(f"expected anchor not found in {path.name}:\n{needle[:160]}")
    path.write_text(source.replace(needle, replacement, 1))
    return True


def undo_token_concatenation() -> bool:
    """Remove the earlier, wrong approach if this repo still carries it."""
    source = PIPELINE.read_text()
    needle = """        # Several views describe ONE object, so fold them into a single conditioning
        # sequence along the token axis: (B, N, D) -> (1, B*N, D). Leaving them batched
        # would ask the sampler for B separate objects, since it builds noise shaped
        # (num_samples, ...) and pairs it with the conditioning batch.
        if cond.ndim == 3 and cond.shape[0] > 1:
            cond = cond.reshape(1, -1, cond.shape[-1])
"""
    if needle not in source:
        return False
    # Views must stay separate so the sampler can run the denoiser per view.
    PIPELINE.write_text(source.replace(needle, "", 1))
    return True


def patch_sampler_injection() -> bool:
    needle = """    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:"""
    replacement = '''    def install_multi_image_hooks(self, num_views: int) -> None:
        """Install the per-view fusion hook on every sampler, once."""
        for name in ('sparse_structure_sampler', 'shape_slat_sampler', 'tex_slat_sampler'):
            sampler = getattr(self, name, None)
            if sampler is None or getattr(sampler, '_multi_image_hooked', False):
                continue
            self.inject_sampler_multi_image(
                name, getattr(self, 'multi_image_mode', 'multidiffusion')
            )
            sampler._multi_image_hooked = True

    def inject_sampler_multi_image(self, sampler_name: str, mode: str = 'multidiffusion'):
        """Combine several views by fusing the denoiser's predictions, not its inputs.

        Concatenating conditioning tokens fails because TRELLIS has no view embeddings
        and cannot tell the views apart. Instead keep the views separate -- conditioning
        stays shaped (B, N, D) -- and slice one view per call, so classifier-free
        guidance and the guidance interval still apply per view exactly as designed.

        Ported from TRELLIS v1, which shipped this and which v2 dropped.
        """
        sampler = getattr(self, sampler_name)
        original = sampler._inference_model
        counter = {'step': 0}

        def per_view(model, x_t, t, cond, **kwargs):
            views = cond.shape[0] if hasattr(cond, 'shape') and cond.ndim == 3 else 1
            if views <= 1:
                return original(model, x_t, t, cond, **kwargs)

            neg_cond = kwargs.pop('neg_cond', None)

            def call(index):
                extra = dict(kwargs)
                if neg_cond is not None:
                    # neg_cond mirrors cond's batch, so slice it in step.
                    extra['neg_cond'] = (
                        neg_cond[index:index + 1]
                        if getattr(neg_cond, 'shape', [1])[0] == views
                        else neg_cond
                    )
                return original(model, x_t, t, cond[index:index + 1], **extra)

            if not getattr(per_view, '_announced', False):
                print(f"MULTIVIEW:: fusing {views} views, mode={mode}", flush=True)
                per_view._announced = True
            per_view._fusions = getattr(per_view, '_fusions', 0) + 1

            if mode == 'stochastic':
                # One view per step, cycling. Deterministic, so runs stay reproducible.
                index = counter['step'] % views
                counter['step'] += 1
                return call(index)

            # multidiffusion: every view predicts, then average.
            predictions = [call(i) for i in range(views)]
            total = predictions[0]
            for prediction in predictions[1:]:
                total = total + prediction
            return total / len(predictions)

        sampler._inference_model = per_view
        sampler._multi_image_per_view = per_view

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:'''
    return replace(PIPELINE, needle, replacement, "install_multi_image_hooks")


def patch_contextmanager_import() -> bool:
    source = PIPELINE.read_text()
    if "from contextlib import contextmanager" in source:
        return False
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(index, "from contextlib import contextmanager\n")
            break
    else:
        raise RuntimeError("no import block found in the pipeline module")
    PIPELINE.write_text("".join(lines))
    return True


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
    if replace(PIPELINE, needle, replacement, "Accept either one image or several"):
        return True
    return False


def patch_run_wraps_samplers() -> bool:
    """Install the hook on every sampler for the whole run.

    A `with` block would only cover the first sampler -- shape and texture SLat are
    sampled in later statements -- so the hook is installed once instead. It is inert
    for a single view (it delegates straight to the original), so it can stay in place.
    """
    needle = """        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure("""
    replacement = """        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        # With several views, fuse per-view predictions inside EVERY sampler rather
        # than merging the conditioning up front. Installed for the whole run, since
        # shape and texture SLat are sampled in later statements.
        self.install_multi_image_hooks(len(images))
        coords = self.sample_sparse_structure("""
    # The method definition itself contains ``install_multi_image_hooks``. Using that
    # broad substring as the idempotence marker made a fresh install skip this call,
    # leaving all of the injected multi-view machinery dormant.
    return replace(
        PIPELINE,
        needle,
        replacement,
        "self.install_multi_image_hooks(len(images))",
    )


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


def patch_generate_mode_arg() -> bool:
    needle = '    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")'
    replacement = (
        '    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")\n'
        '    parser.add_argument(\n'
        '        "--multi-image-mode", choices=("multidiffusion", "stochastic"),\n'
        '        default="multidiffusion",\n'
        '        help="How to fuse several views: average every view\'s prediction each "\n'
        '             "step (multidiffusion), or use one view per step (stochastic)",\n'
        '    )'
    )
    return replace(GENERATE, needle, replacement, "--multi-image-mode")


def patch_generate_load() -> bool:
    needle = """    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")"""
    replacement = """    img = [PILImage.open(path) for path in args.image]
    for path, view in zip(args.image, img):
        print(f"Input: {path} ({view.size[0]}x{view.size[1]})")
    if len(img) == 1:
        img = img[0]
    else:
        print(f"Conditioning on {len(img)} views (mode={args.multi_image_mode})")
        pipeline.multi_image_mode = args.multi_image_mode"""
    return replace(GENERATE, needle, replacement, "pipeline.multi_image_mode =")


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
    if undo_token_concatenation():
        print("  removed  : earlier token-concatenation approach (wrong technique)")
    steps = (
        ("contextlib import", patch_contextmanager_import),
        ("per-view sampler injection", patch_sampler_injection),
        ("run() accepts a view list", patch_run),
        ("run() wraps samplers with the hook", patch_run_wraps_samplers),
        ("generate.py accepts several paths", patch_generate_arg),
        ("generate.py --multi-image-mode", patch_generate_mode_arg),
        ("generate.py loads every view", patch_generate_load),
        ("generate.py existence check", patch_generate_exists),
    )
    for label, step in steps:
        print(f"  {'applied ' if step() else 'already '}: {label}")
    print("multi-view patch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
