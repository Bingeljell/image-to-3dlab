#!/usr/bin/env python3
"""Apply the minimal inference-preserving Mac patch to a Microsoft Space tree.

The target is a clean checkout of the official TRELLIS.2 Hugging Face Space, not
Shiv's patched checkout.  Every edit is narrow, idempotent, and fails closed when
Microsoft's source no longer matches the audited anchors.

This patch deliberately does not alter preprocessing, sampler defaults, latent
normalization, flow equations, or decoder equations.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis-space-mac" / "TRELLIS.2"
ATTENTION_HELPER_SOURCE = Path(__file__).with_name("trellis_attention_compat.py")
ATTENTION_HELPER_TARGET = "trellis2/_assetfurnace_attention.py"
SPARSE_ATTENTION_TARGET = "trellis2/modules/sparse/attention/full_attn.py"
DEVICE_ADAPTER = (
    "        from trellis2._assetfurnace_attention import accelerator_device\n"
    "        device = accelerator_device()"
)
LEGACY_DEVICE = '        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")'


def install_attention_helper(root: Path) -> bool:
    path = root / ATTENTION_HELPER_TARGET
    source = ATTENTION_HELPER_SOURCE.read_bytes()
    if path.is_file() and path.read_bytes() == source:
        return False
    path.write_bytes(source)
    return True


def replace_exact(source: str, old: str, new: str, *, count: int, label: str, previous: str | None = None) -> tuple[str, bool]:
    """Replace an audited anchor or fail instead of producing a partial port."""
    if new in source:
        return source, False
    if previous is not None and previous in source:
        old = previous
    found = source.count(old)
    if found != count:
        raise RuntimeError(f"{label}: expected {count} source anchors, found {found}")
    return source.replace(old, new, count), True


def patch_file(path: Path, transform) -> bool:
    source = path.read_text()
    patched, changed = transform(source)
    if changed:
        path.write_text(patched)
    return changed


def patch_sparse_config(root: Path) -> bool:
    path = root / "trellis2/modules/sparse/config.py"
    backends = "['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa']"
    mlx_backends = backends[:-1] + ", 'mlx']"

    def transform(source: str) -> tuple[str, bool]:
        if mlx_backends in source:
            if (source.count(mlx_backends) != 2 or source.count("Literal" + mlx_backends) != 1
                    or backends in source or "['xformers', 'flash_attn', 'flash_attn_3']" in source
                    or "Literal['xformers', 'flash_attn']" in source):
                raise RuntimeError("sparse attention allow-lists: inconsistent MLX patch")
            return source, False
        changed = False
        source, did = replace_exact(
            source,
            "['xformers', 'flash_attn', 'flash_attn_3']",
            backends,
            count=1,
            label="sparse attention environment allow-list",
        )
        changed |= did
        source, did = replace_exact(
            source,
            "Literal['xformers', 'flash_attn']",
            "Literal" + backends,
            count=1,
            label="sparse attention type allow-list",
        )
        return source, changed | did

    return patch_file(path, transform)


def patch_sparse_attention(root: Path) -> bool:
    path = root / SPARSE_ATTENTION_TARGET
    anchor = '''    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")'''
    start = "    elif config.ATTN == 'sdpa':\n"
    metal_branch = "    elif config.ATTN == 'metal_flash':"
    mlx_branch = "    elif config.ATTN == 'mlx':"
    legacy = "        # image-to-3dlab: evaluate each packed sequence independently with"
    replacement = start + """\
        from trellis2._assetfurnace_attention import chunked_sdpa, unpack_attention
        q, k, v = unpack_attention(num_all_args, 1, locals())
        out = chunked_sdpa(q, k, v, q_seqlen, kv_seqlen)
    elif config.ATTN == 'metal_flash':
        from trellis2._assetfurnace_attention import metal_attention, unpack_attention
        q, k, v = unpack_attention(num_all_args, 1, locals())
        out = metal_attention(q, k, v, q_seqlen, kv_seqlen, device)
"""

    def transform(source: str) -> tuple[str, bool]:
        if legacy in source:
            previous = start + legacy
            if source.count(previous) != 1 or source.count(anchor) != 1:
                raise RuntimeError("fused Metal sparse-attention dispatch: older patch is ambiguous")
            before, _, patched = source.partition(previous)
            old_branches, found, after = patched.partition(anchor)
            if not found:
                raise RuntimeError("fused Metal sparse-attention dispatch: older patch end is missing")
            branches = [
                line for line in old_branches.splitlines()[1:]
                if line.strip() and not line.startswith("        ")
            ]
            if branches not in ([metal_branch], [metal_branch, mlx_branch]):
                raise RuntimeError("fused Metal sparse-attention dispatch: older patch boundaries changed")
            mlx_position = (
                old_branches.index("\n" + mlx_branch + "\n") + 1
                if mlx_branch in branches else len(old_branches)
            )
            return before + replacement + old_branches[mlx_position:] + found + after, True
        installed_with_mlx = replacement + mlx_branch + "\n"
        if installed_with_mlx in source:
            _, _, mlx_body = source.partition(installed_with_mlx)
            if source.count(installed_with_mlx) != 1 or source.count(anchor) != 1 or anchor not in mlx_body:
                raise RuntimeError("fused Metal sparse-attention dispatch: MLX patch boundaries changed")
            return source, False
        return replace_exact(source, anchor, replacement + anchor, count=1,
                             label="fused Metal sparse-attention dispatch")

    changed = patch_file(path, transform)
    return install_attention_helper(root) | changed


def patch_image_feature_extractor(root: Path) -> bool:
    path = root / "trellis2/modules/image_feature_extractor.py"
    lifecycle = '''    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()'''
    device_lifecycle = '''    @property
    def device(self):
        # image-to-3dlab: keep preprocessing tensors beside the DINO model.
        from trellis2._assetfurnace_attention import model_device
        return model_device(self.model)

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        from trellis2._assetfurnace_attention import accelerator_device
        device = accelerator_device()
        self.model.to(device)

    def cpu(self):
        self.model.cpu()'''

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_exact(
            source,
            lifecycle,
            device_lifecycle,
            count=2,
            label="DINO device lifecycle",
            previous=device_lifecycle.replace(DEVICE_ADAPTER, LEGACY_DEVICE).replace(
                "        from trellis2._assetfurnace_attention import model_device\n"
                "        return model_device(self.model)",
                "        return next(self.model.parameters()).device",
            ),
        )
        changed |= did
        if "torch.stack(image).cuda()" in source:
            if source.count("torch.stack(image).cuda()") != 2:
                raise RuntimeError("DINO list input: unexpected number of CUDA anchors")
            source = source.replace("torch.stack(image).cuda()", "torch.stack(image).to(self.device)")
            changed = True
        if "self.transform(image).cuda()" in source:
            if source.count("self.transform(image).cuda()") != 2:
                raise RuntimeError("DINO transform: unexpected number of CUDA anchors")
            source = source.replace("self.transform(image).cuda()", "self.transform(image).to(self.device)")
            changed = True
        source, did = replace_exact(
            source,
            "        for i, layer_module in enumerate(self.model.layer):",
            "        from trellis2._assetfurnace_attention import dino_layers\n"
            "        layers = dino_layers(self.model)\n"
            "        for i, layer_module in enumerate(layers):",
            count=1,
            label="DINOv3 transformer layer compatibility",
            previous="        layers = getattr(getattr(self.model, 'model', self.model), 'layer')\n"
                     "        for i, layer_module in enumerate(layers):",
        )
        return source, changed | did

    return patch_file(path, transform)


def patch_pipeline_device(root: Path) -> bool:
    base = root / "trellis2/pipelines/base.py"
    pipeline = root / "trellis2/pipelines/trellis2_image_to_3d.py"

    def base_transform(source: str) -> tuple[str, bool]:
        return replace_exact(
            source,
            '        self.to(torch.device("cuda"))',
            '        from trellis2._assetfurnace_attention import accelerator_device\n'
            '        device = accelerator_device()\n'
            '        self.to(device)',
            count=1,
            label="pipeline accelerator selection",
            previous=LEGACY_DEVICE + '\n        self.to(device)',
        )

    def pipeline_transform(source: str) -> tuple[str, bool]:
        return replace_exact(
            source,
            "        torch.cuda.empty_cache()",
            "        from trellis2._assetfurnace_attention import empty_cuda_cache\n"
            "        empty_cuda_cache()",
            count=1,
            label="CUDA cache guard",
            previous="        if torch.cuda.is_available():\n            torch.cuda.empty_cache()",
        )

    return patch_file(base, base_transform) | patch_file(pipeline, pipeline_transform)


# The constructor is split so this file never contains the call as one string.
# Upstream writes it on one line; an older patch wrote the same call in a ternary.
_CALL = (
    "getattr(rembg, "
    "args['rembg_model']['name'])(**args['rembg_model']['args'])"
)
CONSTRUCTOR = f"        new_pipeline.rembg_model = {_CALL}"
TERNARY = (
    "        new_pipeline.rembg_model = (\n"
    f"            {_CALL}\n"
    "            if load_rembg else None\n"
    "        )"
)
DISABLED = "        new_pipeline.rembg_model = None"


def pipeline_path(root: Path) -> Path:
    return root / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"


def _remover_is_disabled(source: str) -> bool:
    """Recognize the audited assignment and reject further writes to the remover."""
    if _CALL in source:
        return False
    try:
        tree = ast.parse(dedent(source))
    except SyntaxError:
        return False
    approved = None
    writes = []
    for node in ast.walk(tree):
        match node:
            case ast.Assign(targets=[
                ast.Attribute(value=ast.Name(id="new_pipeline"), attr="rembg_model") as target
            ], value=ast.Constant(value=None)):
                approved = target
            case ast.Attribute(attr="rembg_model", ctx=ast.Store() | ast.Del()):
                writes.append(node)
            case ast.Call(func=ast.Name(id="setattr" | "delattr"),
                          args=[_, ast.Constant(value="rembg_model"), *_]):
                return False
    return approved is not None and writes == [approved]


def is_patched(root: Path) -> bool:
    """True when the only background remover assignment disables it."""
    try:
        text = pipeline_path(root).read_text()
    except (OSError, UnicodeError):
        return False
    return _remover_is_disabled(text)


def has_old_optional_form(root: Path) -> bool:
    """True when the tree carries the constructor behind `if load_rembg`, which an earlier
    version of this patch wrote. Such a tree is ours to upgrade, not a pristine checkout."""
    try:
        return TERNARY in pipeline_path(root).read_text()
    except (OSError, UnicodeError):
        return False


def bria_refusal(root: Path) -> str | None:
    """Why this Space checkout may not run, or None."""
    if is_patched(root):
        return attention_helper_refusal(root)
    return (
        f"{root} still loads BRIA RMBG-2.0, which this repo's "
        "generation pipeline must never load. Apply the guardrail first:\n"
        f"    python scripts/patch_trellis_space_core.py --root {root}"
    )


def attention_helper_refusal(root: Path) -> str | None:
    """Older inline patches need no helper. A linked copy must match this release."""
    attention = root / SPARSE_ATTENTION_TARGET
    try:
        if not attention.is_file() or "from trellis2._assetfurnace_attention import" not in attention.read_text():
            return None
        if (root / ATTENTION_HELPER_TARGET).read_bytes() == ATTENTION_HELPER_SOURCE.read_bytes():
            return None
    except (OSError, UnicodeError):
        pass
    return (
        f"The TRELLIS attention helper in {root} is out of date or missing. Refresh it first:\n"
        f"    python scripts/patch_trellis_space_core.py --root {root}"
    )


def patch_optional_rembg(root: Path) -> bool:
    """Never construct BRIA RMBG-2.0. Callers may still pass load_rembg; it is ignored."""
    path = pipeline_path(root)

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_exact(
            source,
            '    def from_pretrained(path: str) -> "Trellis2ImageTo3DPipeline":',
            '    def from_pretrained(path: str, *, load_rembg: bool = True) -> "Trellis2ImageTo3DPipeline":',
            count=1,
            label="background-removal flag is accepted and ignored",
        )
        changed |= did
        if _CALL not in source:
            if _remover_is_disabled(source):
                return source, changed
            raise RuntimeError(
                "background-removal model disabled: expected 1 source anchors, found 0"
            )
        # Do not use replace_exact here: it treats the None assignment as already
        # done, and would leave a constructor sitting beside it.
        old = TERNARY if TERNARY in source else CONSTRUCTOR
        found = source.count(old)
        if found != 1:
            raise RuntimeError(
                f"background-removal model disabled: expected 1 source anchors, found {found}"
            )
        source = source.replace(old, DISABLED, 1)
        if not _remover_is_disabled(source):
            raise RuntimeError("background-removal model disabled: found another remover assignment")
        return source, True

    return patch_file(path, transform)


def patch_sparse_tensor_cuda_aliases(root: Path) -> bool:
    path = root / "trellis2/modules/sparse/basic.py"
    varlen = '''    def cuda(self) -> 'VarLenTensor':
        new_feats = self.feats.cuda()
        return self.replace(new_feats)'''
    varlen_device = '''    def cuda(self) -> 'VarLenTensor':
        # API-compatible accelerator alias for Apple Silicon.
        from trellis2._assetfurnace_attention import accelerator_device
        device = accelerator_device()
        new_feats = self.feats.to(device)
        return self.replace(new_feats)'''
    sparse = '''    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)'''
    sparse_device = '''    def cuda(self) -> 'SparseTensor':
        # API-compatible accelerator alias for Apple Silicon.
        from trellis2._assetfurnace_attention import accelerator_device
        device = accelerator_device()
        new_feats = self.feats.to(device)
        new_coords = self.coords.to(device)
        return self.replace(new_feats, new_coords)'''

    def transform(source: str) -> tuple[str, bool]:
        source, first = replace_exact(
            source, varlen, varlen_device, count=1, label="VarLenTensor accelerator alias",
            previous=varlen_device.replace(DEVICE_ADAPTER, LEGACY_DEVICE),
        )
        source, second = replace_exact(
            source, sparse, sparse_device, count=1, label="SparseTensor accelerator alias",
            previous=sparse_device.replace(DEVICE_ADAPTER, LEGACY_DEVICE),
        )
        return source, first | second

    return patch_file(path, transform)


def patch_mesh_device(root: Path) -> bool:
    path = root / "trellis2/representations/mesh/base.py"
    cuda_alias = '''    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)'''
    accelerator_alias = '''    def cuda(self, non_blocking=False):
        from trellis2._assetfurnace_attention import accelerator_device
        device = accelerator_device()
        return self.to(device, non_blocking=non_blocking)'''

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_exact(
            source,
            cuda_alias,
            accelerator_alias,
            count=1,
            label="mesh accelerator alias",
            previous=accelerator_alias.replace(DEVICE_ADAPTER, LEGACY_DEVICE),
        )
        changed |= did
        if "self.vertices.cuda()" in source or "self.faces.cuda()" in source:
            if source.count("self.vertices.cuda()") != 3 or source.count("self.faces.cuda()") != 3:
                raise RuntimeError("mesh cleanup: unexpected number of CUDA anchors")
            source = source.replace("self.vertices.cuda()", "self.vertices.to(self.device)")
            source = source.replace("self.faces.cuda()", "self.faces.to(self.device)")
            changed = True
        return source, changed

    return patch_file(path, transform)


def apply(root: Path) -> list[str]:
    if not (root / "trellis2").is_dir():
        raise RuntimeError(f"not a TRELLIS.2 Space source tree: {root}")
    steps = (
        ("sparse attention config", patch_sparse_config),
        ("fused Metal sparse attention", patch_sparse_attention),
        ("DINO device routing", patch_image_feature_extractor),
        ("pipeline accelerator routing", patch_pipeline_device),
        ("background-removal disabled", patch_optional_rembg),
        ("sparse tensor accelerator aliases", patch_sparse_tensor_cuda_aliases),
        ("mesh device routing", patch_mesh_device),
    )
    changed = []
    for label, function in steps:
        if function(root):
            changed.append(label)
            print(f"applied : {label}")
        else:
            print(f"present : {label}")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    apply(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
