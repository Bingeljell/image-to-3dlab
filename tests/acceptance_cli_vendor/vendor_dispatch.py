"""Import a patched fixture package in a fresh interpreter without native libraries."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def unused_kernel(*args, **kwargs):
    raise AssertionError("unexpected attention kernel")


class Tensor:
    shape = (1, 1, 16)

    def __init__(self, label, axes=()):
        self.label = label
        self.axes = axes

    def transpose(self, first, second):
        return Tensor(self.label, self.axes + ((first, second),))


class Packed:
    def unbind(self, dim):
        return tuple(Tensor(f"{name}@{dim}") for name in "qkv")


def aule_kernel(q, k, v, *, causal):
    assert not causal
    assert [x.label for x in (q, k, v)] == ["q@2", "k@2", "v@2"]
    assert all(x.axes == ((1, 2),) for x in (q, k, v))
    return v


def metal_kernel(q, k, v, q_prefix, kv_prefix, q_max, kv_max, scale):
    assert [x.label for x in (q, k, v)] == ["q@1", "k@1", "v@1"]
    assert q_prefix == ([0, 2, 5], "int32", "fixture-device")
    assert kv_prefix == ([0, 4, 5], "int32", "fixture-device")
    assert (q_max, kv_max, scale) == (3, 4, 0.25)
    return v


def mlx_kernel(q, k, v, q_seqlen, kv_seqlen):
    assert [x.label for x in (q, k, v)] == ["q@1", "k@1", "v@1"]
    assert (q_seqlen, kv_seqlen) == ([2, 3], [4, 1])
    return v


def install_native_fakes():
    torch = ModuleType("torch")
    torch.int32 = "int32"
    torch.tensor = lambda values, *, dtype, device: (values, dtype, device)
    functional = ModuleType("torch.nn.functional")
    functional.scaled_dot_product_attention = unused_kernel
    sys.modules.update({"torch": torch, "torch.nn": ModuleType("torch.nn"),
                        "torch.nn.functional": functional})
    sys.modules["aule"] = SimpleNamespace(flash_attention_triton=aule_kernel)
    sys.modules["flex_gemm"] = SimpleNamespace(
        kernels=SimpleNamespace(metal=SimpleNamespace(sparse_attention_fwd=metal_kernel))
    )


def main():
    route, root = sys.argv[1:]
    sys.path.insert(0, root)

    if route == "pixal":
        from pixal3d.pipelines import _assetfurnace_pixal3d_compat as helper
        from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

        result = Pixal3DImageTo3DPipeline.model_names_to_load
    else:
        install_native_fakes()
        if route == "rocm":
            from trellis2.modules.attention import config, full_attn

            config.set_backend("aule")
            output = full_attn.scaled_dot_product_attention(Packed())
        else:
            from trellis2.modules.sparse import config
            from trellis2.modules.sparse.attention import full_attn

            if route == "space-mlx":
                sys.modules["image_to_3dlab.mlx_attention"] = SimpleNamespace(varlen_attention=mlx_kernel)
                config.set_attn_backend("mlx")
            else:
                config.set_attn_backend("metal_flash")
            output = full_attn.scaled_dot_product_attention(
                Packed(), q_seqlen=[2, 3], kv_seqlen=[4, 1],
            )
        # Dispatch must load the helper before the probe looks it up.
        helper = sys.modules["trellis2._assetfurnace_attention"]
        result = [output.label, output.axes]

    assert "image_to_3dlab" not in sys.modules
    print(json.dumps({"helper": str(Path(helper.__file__).resolve()), "result": result}))


if __name__ == "__main__":
    main()
