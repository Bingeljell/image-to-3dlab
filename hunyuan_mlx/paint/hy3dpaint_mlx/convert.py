"""Torch/diffusers safetensors -> MLX module weight loading.

Shared by the parity tests and the eventual convert scripts. Two universal rules:
  * Conv2d weights are NCHW [out,in,kH,kW] in torch, NHWC [out,kH,kW,in] in MLX -> transpose (0,2,3,1).
  * Linear/Norm weights are identical.
Module trees in this package mirror the diffusers names, so loading is a flat key map plus
optional substring renames (e.g. diffusers' ".to_out.0." -> our ".to_out.").
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_unflatten


def load_torch_weights(module, sd, renames=(), verbose=False):
    """Load a torch state_dict (dict of mx.array or numpy) into an MLX module.

    sd: {torch_key: array}. renames: iterable of (old_substr, new_substr).
    After the update, a parameter the checkpoint left out raises KeyError.
    """
    flat = []
    for k, v in sd.items():
        for a, b in renames:
            k = k.replace(a, b)
        arr = v if isinstance(v, mx.array) else mx.array(v)
        if arr.ndim == 4:  # conv weight NCHW -> NHWC
            arr = arr.transpose(0, 2, 3, 1)
        flat.append((k, arr))
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
