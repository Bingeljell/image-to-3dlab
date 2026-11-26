"""Audit the BRIA disablement gate without importing the vendor pipeline."""

from textwrap import dedent

import patch_trellis_space_core as patcher
import pytest


DINO_CLASS = '''\
class Extractor:
    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def encode(self, image):
        batches = torch.stack(image).cuda()
        return self.transform(image).cuda()
'''
MESH = '''\
class Mesh:
    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)

    def cleanup(self):
        return self.vertices.cuda(), self.faces.cuda()
'''


def write_unit(root, relative, source):
    path = root / "trellis2" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


@pytest.fixture
def space_tree(tmp_path):
    from test_patch_trellis_space_core import UPSTREAM

    paths = {
        "config": write_unit(tmp_path, "modules/sparse/config.py", dedent("""\
            from typing import Literal
            ATTN: Literal['xformers', 'flash_attn'] = 'flash_attn'
            BACKENDS = ['xformers', 'flash_attn', 'flash_attn_3']
            """)),
        "attention": write_unit(tmp_path, "modules/sparse/attention/full_attn.py", dedent("""\
            def dispatch():
                if config.ATTN == "flash_attn":
                    return qkv
                else:
                    raise ValueError(f"Unknown attention module: {config.ATTN}")
            """)),
        "dino": write_unit(tmp_path, "modules/image_feature_extractor.py",
                           DINO_CLASS.replace("Extractor", "DinoV2") + "\n"
                           + DINO_CLASS.replace("Extractor", "DinoV3")
                           + "        for i, layer_module in enumerate(self.model.layer):\n"
                           + "            pass\n"),
        "base": write_unit(tmp_path, "pipelines/base.py", dedent("""\
            class Pipeline:
                def cuda(self):
                    self.to(torch.device("cuda"))
            """)),
        "pipeline": write_unit(tmp_path, "pipelines/trellis2_image_to_3d.py",
                               "class Trellis2ImageTo3DPipeline:\n" + UPSTREAM
                               + "        torch.cuda.empty_cache()\n"),
        "sparse": write_unit(tmp_path, "modules/sparse/basic.py", dedent("""\
            class VarLenTensor:
                def cuda(self) -> 'VarLenTensor':
                    new_feats = self.feats.cuda()
                    return self.replace(new_feats)

            class SparseTensor:
                def cuda(self) -> 'SparseTensor':
                    new_feats = self.feats.cuda()
                    new_coords = self.coords.cuda()
                    return self.replace(new_feats, new_coords)
            """)),
        "mesh": write_unit(tmp_path, "representations/mesh/base.py",
                           MESH + "        repair = self.vertices.cuda(), self.faces.cuda()\n"
                           + "        return self.vertices.cuda(), self.faces.cuda()\n"),
    }
    return paths


def test_space_patch_application_is_repeatable_and_keeps_guard(tmp_path, space_tree, monkeypatch, capsys):
    import ast
    import sys

    monkeypatch.setattr(sys, "argv", ["patch", "--root", str(tmp_path)])
    assert patcher.main() == 0
    first = capsys.readouterr().out
    assert first.count("applied :") == len(space_tree)
    assert patcher.is_patched(tmp_path)
    assert "accelerator_device()" in space_tree["base"].read_text()
    assert "empty_cuda_cache()" in space_tree["pipeline"].read_text()
    assert "self.coords.to(device)" in space_tree["sparse"].read_text()
    assert space_tree["dino"].read_text().count("to(self.device)") == 4
    assert "self.vertices.cuda()" not in space_tree["mesh"].read_text()
    assert "chunked_sdpa" in space_tree["attention"].read_text()
    transformed = {name: path.read_text() for name, path in space_tree.items()}
    for name, source in transformed.items():
        ast.parse(source, name)
    assert patcher.main() == 0
    assert capsys.readouterr().out.count("present :") == len(space_tree)
    assert {name: path.read_text() for name, path in space_tree.items()} == transformed


@pytest.mark.parametrize(("script", "anchor", "apply", "message"), [
    ("dino", "torch.stack(image).cuda()", patcher.patch_image_feature_extractor, "DINO list input"),
    ("dino", "self.transform(image).cuda()", patcher.patch_image_feature_extractor, "DINO transform"),
    ("mesh", "self.faces.cuda()", patcher.patch_mesh_device, "mesh cleanup"),
])
def test_space_patch_does_not_save_when_an_audited_anchor_count_changes(
    tmp_path, space_tree, script, anchor, apply, message,
):
    path = space_tree[script]
    changed = path.read_text().replace(anchor, "changed_upstream()", 1)
    path.write_text(changed)
    with pytest.raises(RuntimeError, match=message):
        apply(tmp_path)
    assert path.read_text() == changed


def test_apply_refuses_directory_without_checkout(tmp_path):
    with pytest.raises(RuntimeError, match="not a TRELLIS.2 Space source tree"):
        patcher.apply(tmp_path)



@pytest.mark.parametrize("missing", [False, True])
def test_space_refuses_wrong_attention_helper_until_patch_repairs_it(tmp_path, space_tree, missing):
    patcher.apply(tmp_path)
    assert patcher.bria_refusal(tmp_path) is None
    helper = tmp_path / patcher.ATTENTION_HELPER_TARGET
    if missing:
        helper.unlink()
    else:
        helper.write_bytes(b"# stale helper\n")

    assert patcher.is_patched(tmp_path)
    refusal = patcher.bria_refusal(tmp_path)
    assert refusal is not None and "attention helper" in refusal
    assert "patch_trellis_space_core.py" in refusal

    assert patcher.apply(tmp_path) == ["fused Metal sparse attention"]
    assert helper.read_bytes() == patcher.ATTENTION_HELPER_SOURCE.read_bytes()
    assert patcher.bria_refusal(tmp_path) is None
    assert patcher.apply(tmp_path) == []
