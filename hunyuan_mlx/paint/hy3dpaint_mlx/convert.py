"""Torch/diffusers safetensors -> MLX module weight loading.

Shared by the parity tests and the eventual convert scripts. Two universal rules:
  * Conv2d weights are NCHW [out,in,kH,kW] in torch, NHWC [out,kH,kW,in] in MLX -> transpose (0,2,3,1).
  * Linear/Norm weights are identical.
Module trees in this package mirror the diffusers names, so loading is a flat key map plus
optional substring renames (e.g. diffusers' ".to_out.0." -> our ".to_out.").
"""

from __future__ import annotations


def remap_parameters(sd, renames=(), *, as_array=None):
    """Yield renamed parameters with convolution weights in channels-last order."""
    for key, value in sd.items():
        for old, new in renames:
            key = key.replace(old, new)
        arr = as_array(value) if as_array is not None else value
        if arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
        yield key, arr


def load_torch_weights(module, sd, renames=(), verbose=False):
    """Load a torch state_dict (dict of mx.array or numpy) into an MLX module.

    sd: {torch_key: array}. renames: iterable of (old_substr, new_substr).
    After the update, a parameter the checkpoint left out raises KeyError.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    def as_mlx(value):
        return value if isinstance(value, mx.array) else mx.array(value)

    flat = list(remap_parameters(sd, renames, as_array=as_mlx))
    module.update(tree_unflatten(flat))
    mx.eval(module.parameters())
    loaded = {key for key, _value in flat}
    missing = sorted(set(parameter_names(module.parameters())) - loaded)
    if missing:
        raise KeyError(f"{len(missing)} missing parameter names, e.g. {missing[:6]}")
    if verbose:
        print(f"loaded {len(flat)} tensors")
    return module


def parameter_names(tree):
    """Dotted names of every leaf in a parameter tree."""
    names = []

    def walk(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                walk(value, path)
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                path = f"{prefix}.{index}" if prefix else str(index)
                walk(value, path)
            return
        if prefix:
            names.append(prefix)

    walk(tree, "")
    return names
