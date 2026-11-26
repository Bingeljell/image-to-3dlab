"""Public patch commands install working imports into offline vendor fixtures."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import patch_trellis_mlx_attention as mlx
import pytest
import trellis_space_generate as space_launcher
from test_patch_pixal3d_model_subset import ALL_MODELS
from test_patch_pixal3d_model_subset import ORIGINAL as PIXAL_PIPELINE
from test_patch_trellis_rocm import UPSTREAM_SOURCES as ROCM_SOURCES
from test_patch_trellis_space_core import UPSTREAM as SPACE_PIPELINE

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
PROBE = Path(__file__).with_name("acceptance_cli_vendor") / "vendor_dispatch.py"
ATTENTION_HELPER = "trellis2/_assetfurnace_attention.py"
SPACE_CONFIG = "trellis2/modules/sparse/config.py"
SPACE_ATTN = "trellis2/modules/sparse/attention/full_attn.py"
SPACE_MODEL = "trellis2/pipelines/trellis2_image_to_3d.py"
PIXAL_MODEL = "pixal3d/pipelines/pixal3d_image_to_3d.py"
PIXAL_HELPER = "pixal3d/pipelines/_assetfurnace_pixal3d_compat.py"
pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="Vendor acceptance is assigned to Linux"
)

# Minimal host code around the pinned anchors used by the existing patch tests. The
# dispatch clauses being tested come only from running the real patch commands below.
DINO_LIFECYCLE = """    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
"""
SPACE_SOURCES = {
    SPACE_CONFIG: ROCM_SOURCES[f"TRELLIS.2/{SPACE_CONFIG}"],
    SPACE_ATTN: ROCM_SOURCES[f"TRELLIS.2/{SPACE_ATTN}"].replace(
        "from ... import config", "from .. import config",
    ).replace(
        "    num_all_args = 1 if kv is None else 2",
        "    from types import SimpleNamespace\n"
        "    s = SimpleNamespace(replace=lambda out: out)\n"
        "    device = 'fixture-device'\n"
        "    num_all_args = 1 if kv is None else 2",
    ),
    SPACE_MODEL: "class Trellis2ImageTo3DPipeline:\n" + SPACE_PIPELINE
    + "        torch.cuda.empty_cache()\n",
    "trellis2/pipelines/base.py": 'class Pipeline:\n    def cuda(self):\n'
    '        self.to(torch.device("cuda"))\n',
    "trellis2/modules/image_feature_extractor.py": "class DINOv2:\n" + DINO_LIFECYCLE
    + "\nclass DINOv3:\n" + DINO_LIFECYCLE
    + "\n    def embed(self):\n        for i, layer_module in enumerate(self.model.layer):\n"
    "            pass\n",
    "trellis2/modules/sparse/basic.py": """class VarLenTensor:
    def cuda(self) -> 'VarLenTensor':
        new_feats = self.feats.cuda()
        return self.replace(new_feats)

class SparseTensor:
    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)
""",
    "trellis2/representations/mesh/base.py": """class Mesh:
    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)
""",
}


def write_tree(root, files):
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    # Package imports must resolve to this checkout, never installed vendor modules.
    for package in ("trellis2", "pixal3d"):
        path = next(root.glob(f"**/{package}"), None)
        if path is not None:
            (path / "__init__.py").touch()


def hashes(root):
    return {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def external_cwd(tmp_path):
    path = tmp_path / "ailleurs"
    path.mkdir()
    return path


def python(cwd, *args, **env):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=cwd,
        env={"PATH": os.defpath, "HOME": str(cwd), "PYTHONDONTWRITEBYTECODE": "1", **env},
        capture_output=True,
        text=True,
        timeout=15,
    )


def patch(cwd, script, *args):
    process = python(cwd, SCRIPTS / script, *args)
    assert process.returncode == 0, process.stdout + process.stderr
    return process.stdout


def assert_dispatch(cwd, route, root, helper, expected, **env):
    process = python(cwd, PROBE, route, root, **env)
    assert process.returncode == 0, process.stderr
    observed = json.loads(process.stdout.splitlines()[-1])
    assert observed == {"helper": str(root / helper), "result": expected}


def test_rocm_patch_installs_and_repairs_working_attention_helper(tmp_path, external_cwd):
    root = tmp_path / "ROCm checkout"
    dense_attention = "TRELLIS.2/trellis2/modules/attention/full_attn.py"
    write_tree(root, ROCM_SOURCES | {
        dense_attention: ROCM_SOURCES[dense_attention].replace("from .. import config",
                                                               "from . import config"),
    })
    checkout = root / "TRELLIS.2"
    script = "patch_trellis_rocm.py"
    helper = checkout / ATTENTION_HELPER
    patch(external_cwd, script, "--root", root)

    installed = hashes(root)
    assert helper.read_bytes() == (SCRIPTS / "trellis_attention_compat.py").read_bytes()
    assert_dispatch(external_cwd, "rocm", checkout, ATTENTION_HELPER, ["v@2", [[1, 2], [1, 2]]])
    patch(external_cwd, script, "--root", root)
    assert hashes(root) == installed

    helper.unlink()
    refused = python(external_cwd, SCRIPTS / script, "--root", root, "--check")
    assert refused.returncode == 1
    assert "not all patches applied" in refused.stdout
    assert not helper.exists()
    patch(external_cwd, script, "--root", root)
    assert hashes(root) == installed
    patch(external_cwd, script, "--root", root, "--check")


def test_pixal_patch_installs_working_checkpoint_selection(tmp_path, external_cwd):
    root = tmp_path / "Pixal checkout"
    target = root / PIXAL_MODEL
    script = "patch_pixal3d_model_subset.py"
    write_tree(root, {PIXAL_MODEL: PIXAL_PIPELINE})
    patch(external_cwd, script, target)

    assert (root / PIXAL_HELPER).read_bytes() == (SCRIPTS / "pixal3d_compat.py").read_bytes()
    assert_dispatch(external_cwd, "pixal", root, PIXAL_HELPER, ALL_MODELS[:-1],
                    PIXAL3D_SKIP_MODELS="tex_slat_decoder")
    installed = hashes(root)
    patch(external_cwd, script, target)
    assert hashes(root) == installed

    # The same CLI must refuse an upstream layout that no longer matches the anchor.
    refused_root = tmp_path / "changed Pixal checkout"
    write_tree(refused_root, {PIXAL_MODEL: PIXAL_PIPELINE.replace(
        "        'tex_slat_decoder',\n", "        'other_decoder',\n",
    )})
    before = hashes(refused_root)
    process = python(external_cwd, SCRIPTS / script, refused_root / PIXAL_MODEL)
    assert process.returncode != 0
    assert "anchor not found or changed" in process.stderr
    assert hashes(refused_root) == before


def test_space_patch_installs_working_attention_and_disables_bria(
    tmp_path, external_cwd, monkeypatch,
):
    vendor = tmp_path / "Space checkout"
    checkout = vendor / "TRELLIS.2"
    script = "patch_trellis_space_core.py"
    write_tree(checkout, SPACE_SOURCES)
    patch(external_cwd, script, "--root", checkout)
    assert (checkout / ATTENTION_HELPER).read_bytes() == (
        SCRIPTS / "trellis_attention_compat.py"
    ).read_bytes()
    installed = hashes(checkout)
    patch(external_cwd, script, "--root", checkout)
    assert hashes(checkout) == installed
    assert_dispatch(external_cwd, "space", checkout, ATTENTION_HELPER, ["v@1", []])
    assert space_launcher.bria_refusal(vendor) is None

    pipeline = checkout / SPACE_MODEL
    # A second constructor beside the disabled assignment still blocks launch.
    with pipeline.open("a", encoding="utf-8") as out:
        out.write(SPACE_PIPELINE.splitlines(keepends=True)[-1])
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    snapshot = tmp_path / "cache/models--microsoft--TRELLIS.2-4B/snapshots/fixture"
    snapshot.mkdir(parents=True)
    (snapshot / "model.marker").touch()

    def unexpected_start(*args):
        pytest.fail("TRELLIS environment startup ran before the BRIA guard")

    monkeypatch.setattr(space_launcher, "configure_environment", unexpected_start)
    with pytest.raises(SystemExit, match="still loads BRIA RMBG-2.0"):
        space_launcher.load_pipeline(vendor, "sdpa", load_rembg=False)


def test_legacy_mlx_upgrade_survives_space_patch_reapplication(tmp_path, external_cwd):
    checkout = tmp_path / "TRELLIS.2"
    space_script = "patch_trellis_space_core.py"
    mlx_script = "patch_trellis_mlx_attention.py"
    write_tree(checkout, SPACE_SOURCES)
    # The Space patch provides the backend allow-list expected by the old MLX add-on.
    patch(external_cwd, space_script, "--root", checkout)
    config = checkout / SPACE_CONFIG
    dispatcher = checkout / SPACE_ATTN
    config.write_text(
        config.read_text(encoding="utf-8").replace(mlx.BACKEND_LIST_OLD, mlx.BACKEND_LIST_NEW),
        encoding="utf-8",
    )
    legacy_dispatcher = dispatcher.read_text(encoding="utf-8").replace(
        mlx.DISPATCH_ANCHOR, mlx.LEGACY_DISPATCH,
    )
    dispatcher.write_text(legacy_dispatcher, encoding="utf-8")

    legacy = hashes(checkout)
    patch(external_cwd, space_script, "--root", checkout)
    assert hashes(checkout) == legacy

    patch(external_cwd, mlx_script, "--root", checkout)
    assert config.read_text(encoding="utf-8").count(mlx.BACKEND_LIST_NEW) == 2
    dispatcher_text = dispatcher.read_text(encoding="utf-8")
    assert mlx.LEGACY_DISPATCH not in dispatcher_text
    assert 'from trellis2._assetfurnace_attention import mlx_sparse_attention' in dispatcher_text
    upgraded = hashes(checkout)
    helper = checkout / ATTENTION_HELPER
    helper.write_text("# stale helper\n", encoding="utf-8")
    patch(external_cwd, space_script, "--root", checkout)
    assert hashes(checkout) == upgraded
    patch(external_cwd, space_script, "--root", checkout)
    patch(external_cwd, mlx_script, "--root", checkout)
    assert hashes(checkout) == upgraded
    assert space_launcher.bria_refusal(tmp_path) is None

    # The helper locates its Python package via an ancestor of the vendor checkout.
    mlx_module = tmp_path / "image_to_3dlab" / "mlx_attention.py"
    mlx_module.parent.mkdir()
    mlx_module.touch()
    assert_dispatch(external_cwd, "space", checkout, ATTENTION_HELPER, ["v@1", []])
    assert_dispatch(external_cwd, "space-mlx", checkout, ATTENTION_HELPER, ["v@1", []])
