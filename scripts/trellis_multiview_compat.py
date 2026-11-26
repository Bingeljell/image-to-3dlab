"""Keep TRELLIS views separate and fuse their denoiser predictions."""

from os.path import exists

import torch


def report_missing_inputs(paths):
    """Report missing paths in the order the generator received them."""
    missing = [path for path in paths if not exists(path)]
    if missing:
        print(f"Error: {', '.join(missing)} not found")
    return bool(missing)


def load_input_views(paths, pipeline, mode):
    """Open generator inputs and select single-view or multi-view dispatch."""
    from PIL import Image

    views = [Image.open(path) for path in paths]
    for path, view in zip(paths, views):
        print(f"Input: {path} ({view.size[0]}x{view.size[1]})")
    if len(views) == 1:
        return views[0]
    print(f"Conditioning on {len(views)} views (mode={mode})")
    pipeline.multi_image_mode = mode
    return views


def prepare_conditioning(pipeline, image, preprocess_image, seed, pipeline_type):
    """Prepare one conditioning batch per resolution, keeping views separate."""
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if preprocess_image:
        images = [pipeline.preprocess_image(view) for view in images]
    torch.manual_seed(seed)
    low = pipeline.get_cond(images, 512)
    high = pipeline.get_cond(images, 1024) if pipeline_type != "512" else None
    return images, low, high


def install_multi_image_hooks(pipeline):
    """Install the fusion hook once on every sampler the pipeline has."""
    for name in ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"):
        sampler = getattr(pipeline, name, None)
        if sampler is None or getattr(sampler, "_multi_image_hooked", False):
            continue
        pipeline_mode = getattr(pipeline, "multi_image_mode", "multidiffusion")
        pipeline.inject_sampler_multi_image(name, pipeline_mode)
        sampler._multi_image_hooked = True


def inject_sampler_multi_image(sampler, mode="multidiffusion"):
    """Apply guidance per view and combine the resulting denoiser predictions."""
    original = sampler._inference_model
    step = 0

    def per_view(model, x_t, t, cond, **kwargs):
        nonlocal step
        views = cond.shape[0] if hasattr(cond, "shape") and cond.ndim == 3 else 1
        if views <= 1:
            return original(model, x_t, t, cond, **kwargs)

        neg_cond = kwargs.pop("neg_cond", None)

        def call(index):
            extra = dict(kwargs)
            if neg_cond is not None:
                extra["neg_cond"] = (
                    neg_cond[index:index + 1]
                    if getattr(neg_cond, "shape", [1])[0] == views
                    else neg_cond
                )
            return original(model, x_t, t, cond[index:index + 1], **extra)

        if not getattr(per_view, "_announced", False):
            print(f"MULTIVIEW:: fusing {views} views, mode={mode}", flush=True)
            per_view._announced = True
        per_view._fusions = getattr(per_view, "_fusions", 0) + 1

        if mode == "stochastic":
            index = step % views
            step += 1
            return call(index)

        predictions = [call(i) for i in range(views)]
        total = predictions[0]
        for prediction in predictions[1:]:
            total = total + prediction
        return total / len(predictions)

    sampler._inference_model = per_view
    sampler._multi_image_per_view = per_view
