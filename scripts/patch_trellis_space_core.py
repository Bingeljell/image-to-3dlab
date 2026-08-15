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
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis-space-mac" / "TRELLIS.2"


def replace_exact(source: str, old: str, new: str, *, count: int, label: str) -> tuple[str, bool]:
    """Replace an audited anchor or fail instead of producing a partial port."""
    if new in source:
        return source, False
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

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_exact(
            source,
            "['xformers', 'flash_attn', 'flash_attn_3']",
            "['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa']",
            count=1,
            label="sparse attention environment allow-list",
        )
        changed |= did
        source, did = replace_exact(
            source,
            "Literal['xformers', 'flash_attn']",
            "Literal['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa']",
            count=1,
            label="sparse attention type allow-list",
        )
        return source, changed | did

    return patch_file(path, transform)


def patch_sparse_attention(root: Path) -> bool:
    path = root / "trellis2/modules/sparse/attention/full_attn.py"
    anchor = '''    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")'''
    replacement = '''    elif config.ATTN == 'sdpa':
        # image-to-3dlab: evaluate each packed sequence independently with
        # PyTorch SDPA.  This preserves FlashAttention's block-diagonal
        # semantics without padding or a dense cross-sequence mask.
        #
        # MPS has no flash/memory-efficient SDPA kernel, so PyTorch materialises
        # the full [H, Lq, Lkv] score tensor.  At TRELLIS.2-4B's real Stage-3
        # shape (22,894 tokens, 12 heads, fp32) that single buffer is ~25 GiB
        # and aborts the Metal allocator on a 32 GiB machine.  Chunking the
        # query axis bounds the score tensor to [H, chunk, Lkv] and is
        # numerically exact, because softmax runs per query over all keys.
        # Override the block size with SDPA_Q_CHUNK; the default caps the
        # self-attention score buffer near ~1 GiB.
        import os as _os
        import torch.nn.functional as F
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)

        q_chunk = int(_os.environ.get("SDPA_Q_CHUNK", "1024"))
        out_parts = []
        q_offset = 0
        kv_offset = 0
        for q_length, kv_length in zip(q_seqlen, kv_seqlen):
            k_part = k[kv_offset:kv_offset + kv_length].permute(1, 0, 2).unsqueeze(0)
            v_part = v[kv_offset:kv_offset + kv_length].permute(1, 0, 2).unsqueeze(0)
            seq_parts = []
            for start in range(0, q_length, q_chunk):
                stop = min(start + q_chunk, q_length)
                q_part = q[q_offset + start:q_offset + stop].permute(1, 0, 2).unsqueeze(0)
                part = F.scaled_dot_product_attention(q_part, k_part, v_part)
                seq_parts.append(part.squeeze(0).permute(1, 0, 2))
            if seq_parts:
                out_parts.append(torch.cat(seq_parts, dim=0))
            q_offset += q_length
            kv_offset += kv_length
        out = torch.cat(out_parts, dim=0)
    elif config.ATTN == 'metal_flash':
        # image-to-3dlab: Pedro's fused Metal kernel implements the same packed,
        # variable-length equation used by Microsoft's FlashAttention path for
        # head dimensions up to 64. TRELLIS.2-4B uses 128-wide heads, where
        # Pedro silently falls back to an unusably slow serial kernel.
        import math
        import flex_gemm
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)

        if q.shape[-1] > 64:
            raise RuntimeError(
                f"metal_flash supports fast heads only through 64 values; got {q.shape[-1]}. "
                "Use the sdpa backend for TRELLIS.2-4B."
            )

        q_prefix = [0]
        for length in q_seqlen:
            q_prefix.append(q_prefix[-1] + length)
        kv_prefix = [0]
        for length in kv_seqlen:
            kv_prefix.append(kv_prefix[-1] + length)
        cu_seqlens_q = torch.tensor(q_prefix, dtype=torch.int32).to(device)
        cu_seqlens_kv = torch.tensor(kv_prefix, dtype=torch.int32).to(device)
        out = flex_gemm.kernels.metal.sparse_attention_fwd(
            q, k, v,
            cu_seqlens_q, cu_seqlens_kv,
            max(q_seqlen), max(kv_seqlen),
            1.0 / math.sqrt(q.shape[-1]),
        )
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")'''

    def transform(source: str) -> tuple[str, bool]:
        return replace_exact(
            source,
            anchor,
            replacement,
            count=1,
            label="fused Metal sparse-attention dispatch",
        )

    return patch_file(path, transform)


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
        return next(self.model.parameters()).device

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
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
            "        layers = getattr(getattr(self.model, 'model', self.model), 'layer')\n"
            "        for i, layer_module in enumerate(layers):",
            count=1,
            label="DINOv3 transformer layer compatibility",
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
            '        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")\n'
            '        self.to(device)',
            count=1,
            label="pipeline accelerator selection",
        )

    def pipeline_transform(source: str) -> tuple[str, bool]:
        return replace_exact(
            source,
            "        torch.cuda.empty_cache()",
            "        if torch.cuda.is_available():\n            torch.cuda.empty_cache()",
            count=1,
            label="CUDA cache guard",
        )

    return patch_file(base, base_transform) | patch_file(pipeline, pipeline_transform)


def patch_optional_rembg(root: Path) -> bool:
    """Avoid loading the gated remover when an input already carries transparency."""
    path = root / "trellis2/pipelines/trellis2_image_to_3d.py"

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_exact(
            source,
            '    def from_pretrained(path: str) -> "Trellis2ImageTo3DPipeline":',
            '    def from_pretrained(path: str, *, load_rembg: bool = True) -> "Trellis2ImageTo3DPipeline":',
            count=1,
            label="optional background-removal model signature",
        )
        changed |= did
        source, did = replace_exact(
            source,
            "        new_pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])",
            "        new_pipeline.rembg_model = (\n"
            "            getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
            "            if load_rembg else None\n"
            "        )",
            count=1,
            label="optional background-removal model load",
        )
        return source, changed | did

    return patch_file(path, transform)


def patch_sparse_tensor_cuda_aliases(root: Path) -> bool:
    path = root / "trellis2/modules/sparse/basic.py"
    varlen = '''    def cuda(self) -> 'VarLenTensor':
        new_feats = self.feats.cuda()
        return self.replace(new_feats)'''
    varlen_device = '''    def cuda(self) -> 'VarLenTensor':
        # API-compatible accelerator alias for Apple Silicon.
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        new_feats = self.feats.to(device)
        return self.replace(new_feats)'''
    sparse = '''    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)'''
    sparse_device = '''    def cuda(self) -> 'SparseTensor':
        # API-compatible accelerator alias for Apple Silicon.
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        new_feats = self.feats.to(device)
        new_coords = self.coords.to(device)
        return self.replace(new_feats, new_coords)'''

    def transform(source: str) -> tuple[str, bool]:
        source, first = replace_exact(
            source, varlen, varlen_device, count=1, label="VarLenTensor accelerator alias"
        )
        source, second = replace_exact(
            source, sparse, sparse_device, count=1, label="SparseTensor accelerator alias"
        )
        return source, first | second

    return patch_file(path, transform)


def patch_mesh_device(root: Path) -> bool:
    path = root / "trellis2/representations/mesh/base.py"
    cuda_alias = '''    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)'''
    accelerator_alias = '''    def cuda(self, non_blocking=False):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        return self.to(device, non_blocking=non_blocking)'''

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_exact(
            source,
            cuda_alias,
            accelerator_alias,
            count=1,
            label="mesh accelerator alias",
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
        ("optional background-removal loading", patch_optional_rembg),
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
