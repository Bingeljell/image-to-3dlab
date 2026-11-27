"""scripts/patch_hunyuan_rocm.py: what it changes, and what it refuses to do.

The patches are the whole reason this route works on AMD, and a patch that half-applied
leaves a checkout that imports and then dies minutes into a generation. So the tests cover
the two failure modes that matter: an anchor that has moved, and a second run.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import patch_hunyuan_rocm as gen

# The upstream lines, as they stand at the pinned commit, so the anchors are exercised
# against real text rather than a shape that happens to match itself.
SCHEDULER_SOURCE = """\
class FlowMatch:
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()


class DDIM:
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        indices = (schedule_timesteps == timestep).nonzero()
        return indices[0].item()
"""

ATTN_SOURCE = """\
import torch
import torch.nn.functional as F
from einops import rearrange


def compute(hidden_states):
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        return hidden_states


def per_pbr(self, pbr_hs):
            processed_hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c").to("cuda:0")
            return processed_hs
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "Hunyuan3D-2.1"
    for relative, text in ((gen.SCHEDULER_FILE, SCHEDULER_SOURCE),
                           (gen.ATTN_FILE, ATTN_SOURCE)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_a_fresh_checkout_is_patched(checkout: Path):
    assert gen.is_patched(checkout) is False
    assert gen.patch(checkout) == 2
    assert gen.is_patched(checkout) is True


def test_patching_twice_changes_nothing_and_stays_patched(checkout: Path):
    """The bootstrap may run this on every setup, including an update over an existing one."""
    gen.patch(checkout)
    assert gen.patch(checkout) == 0
    assert gen.is_patched(checkout) is True


def test_patching_ten_times_does_not_stack_the_imports(checkout: Path):
    """The import patch keeps its own anchor in its replacement, so checking the anchor
    before the proof made it prepend `import os` again on every single run."""
    for _ in range(10):
        gen.patch(checkout)
    text = (checkout / gen.ATTN_FILE).read_text()
    assert text.count("from ._i2l_rocm_compat import chunked_attention") == 1
    # This patch keeps its anchor by design, so exactly one of those must survive too.
    assert text.count(gen.ATTN_IMPORT_NEEDLE) == 1
    # The attention call now goes through the shared helper.
    assert gen.ATTN_NEEDLE not in text
    assert text.count("hidden_states = chunked_attention(") == 1


@pytest.mark.posix
def test_patch_preserves_a_symlink_and_target_mode(checkout: Path):
    scheduler = checkout / gen.SCHEDULER_FILE
    real_source = scheduler.with_name("linked_scheduler.py")
    scheduler.rename(real_source)
    scheduler.symlink_to(real_source.name)
    real_source.chmod(0o751)

    assert gen.patch(checkout) == 2
    assert gen.is_patched(checkout)
    assert scheduler.is_symlink()
    assert stat.S_IMODE(real_source.stat().st_mode) == 0o751


def test_the_patched_file_is_still_valid_python(checkout: Path):
    """A patch that produces a syntax error is worse than no patch: it fails on import,
    minutes into setup, with a line number that points at upstream rather than at us."""
    import ast

    gen.patch(checkout)
    for relative in (gen.SCHEDULER_FILE, gen.ATTN_FILE):
        ast.parse((checkout / relative).read_text())


def test_the_scheduler_stops_matching_on_exact_float_equality(checkout: Path):
    """The crash is `indices[pos]` on an empty tensor, minutes into a generation."""
    gen.patch(checkout)
    text = (checkout / gen.SCHEDULER_FILE).read_text()
    assert "(schedule_timesteps == timestep).nonzero()" not in text
    assert "nearest_timestep_indices(schedule_timesteps, timestep)" in text
    # Both schedulers had the line, so both are patched.
    assert text.count(gen.SCHEDULER_REPLACEMENT) == 2


def test_the_scheduler_patch_still_works_when_only_one_scheduler_moved(tmp_path: Path):
    """Replacing is deliberate: the line is in two schedulers and only one may change."""
    root = tmp_path / "Hunyuan3D-2.1"
    path = root / gen.SCHEDULER_FILE
    path.parent.mkdir(parents=True)
    path.write_text(SCHEDULER_SOURCE.split("class DDIM:")[0])
    with pytest.raises(RuntimeError, match="expected 2 occurrence"):
        gen.patch(root)


def _chunked_attention():
    torch = pytest.importorskip("torch")
    from scripts.hunyuan_rocm_compat import chunked_attention

    return chunked_attention, torch


def test_chunked_attention_is_the_same_answer_as_unchunked():
    """The claim this patch makes is "same maths, done in pieces". Check it, for chunk
    sizes around every awkward boundary: smaller than the batch, one, and larger."""
    chunked, torch = _chunked_attention()
    batch, heads, q_len, dim = 8, 4, 37, 16
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(batch, heads, q_len, dim, generator=generator)
    key = torch.randn(batch, heads, 61, dim, generator=generator)
    value = torch.randn(batch, heads, 61, dim, generator=generator)

    unchunked = chunked(query, key, value)
    assert unchunked.shape == (batch, heads, q_len, dim)

    was = os.environ.pop("HY3D_SDPA_QUERY_CHUNK", None)
    try:
        for chunk in (1, 5, 36, 37, 100):
            os.environ["HY3D_SDPA_QUERY_CHUNK"] = str(chunk)
            torch.testing.assert_close(chunked(query, key, value), unchunked,
                                       rtol=1e-5, atol=1e-6,
                                       msg=f"chunk {chunk} disagrees with the whole thing")
    finally:
        os.environ.pop("HY3D_SDPA_QUERY_CHUNK", None)
        if was is not None:
            os.environ["HY3D_SDPA_QUERY_CHUNK"] = was


def test_chunked_attention_matches_with_a_mask_too():
    """Slicing the queries but not the mask is the failure mode that produces
    plausible-looking but wrong textures, so it gets its own check."""
    chunked, torch = _chunked_attention()
    batch, heads, q_len, k_len, dim = 6, 4, 23, 29, 8
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(batch, heads, q_len, dim, generator=generator)
    key = torch.randn(batch, heads, k_len, dim, generator=generator)
    value = torch.randn(batch, heads, k_len, dim, generator=generator)
    # Hide three key positions from every query, which is the case a stale mask gets wrong.
    mask = torch.zeros(batch, 1, q_len, k_len)
    mask[:, :, :, 3:6] = float("-inf")

    whole = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask.expand(batch, heads, q_len, k_len))

    os.environ["HY3D_SDPA_QUERY_CHUNK"] = "4"
    try:
        chunked_result = chunked(query, key, value, mask.expand(batch, heads, q_len, k_len))
    finally:
        os.environ.pop("HY3D_SDPA_QUERY_CHUNK")
    torch.testing.assert_close(chunked_result, whole, rtol=1e-5, atol=1e-6)


def test_a_stale_mask_would_actually_have_been_caught():
    """Guard against the check above passing for the wrong reason: attending with the
    un-sliced mask really does fail loudly, so the equality above means something."""
    chunked, torch = _chunked_attention()
    batch, heads, q_len, k_len, dim = 6, 4, 23, 29, 8
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(batch, heads, q_len, dim, generator=generator)
    key = torch.randn(batch, heads, k_len, dim, generator=generator)
    value = torch.randn(batch, heads, k_len, dim, generator=generator)
    mask = torch.zeros(batch, 1, q_len, k_len)
    mask[:, :, :, 3:6] = float("-inf")
    mask = mask.expand(batch, heads, q_len, k_len)

    os.environ["HY3D_SDPA_QUERY_CHUNK"] = "4"
    try:
        with pytest.raises(RuntimeError, match="must match"):
            torch.cat([torch.nn.functional.scaled_dot_product_attention(
                query[..., i:i + 4, :], key, value, attn_mask=mask)  # whole mask, sliced tokens
                for i in range(0, q_len, 4)], dim=-2)
        assert chunked(query, key, value, mask).shape == query.shape
    finally:
        os.environ.pop("HY3D_SDPA_QUERY_CHUNK")


def test_the_batch_is_left_alone_so_the_pieces_cannot_be_welded_end_to_end():
    """query, key and value are (batch, heads, sequence, head_dim). Slicing dim 0 would
    leave key and value at full batch, SDPA would broadcast the output batch back up to
    match, and cat would glue the chunks into one long sequence."""
    chunked, torch = _chunked_attention()
    batch, heads, q_len, k_len, dim = 3, 2, 19, 13, 8
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(batch, heads, q_len, dim, generator=generator)
    key = torch.randn(batch, heads, k_len, dim, generator=generator)
    value = torch.randn(batch, heads, k_len, dim, generator=generator)

    os.environ["HY3D_SDPA_QUERY_CHUNK"] = "4"
    try:
        result = chunked(query, key, value)
    finally:
        os.environ.pop("HY3D_SDPA_QUERY_CHUNK")
    assert result.shape == query.shape, "the batch or head count moved"
    torch.testing.assert_close(
        result, torch.nn.functional.scaled_dot_product_attention(query, key, value),
        rtol=1e-5, atol=1e-6)


def test_the_hardcoded_device_follows_the_tensor_instead(checkout: Path):
    """.to("cuda:0") resolves under ROCm, so this is not what breaks the run -- it is what
    makes CPU offload, the other lever on a small card, silently do nothing."""
    gen.patch(checkout)
    text = (checkout / gen.ATTN_FILE).read_text()
    assert 'l c").to("cuda:0")' not in text
    assert 'l c")' in text


def snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_an_anchor_that_moved_fails_loudly_instead_of_half_applying(checkout: Path):
    path = checkout / gen.ATTN_FILE
    path.write_text(path.read_text().replace(
        "F.scaled_dot_product_attention(", "F._sdpa_renamed("))
    before = snapshot(checkout)
    with pytest.raises(RuntimeError, match="anchor for this patch is gone"):
        gen.patch(checkout)
    assert snapshot(checkout) == before


def test_a_missing_later_file_leaves_the_checkout_unchanged(checkout: Path):
    (checkout / gen.ATTN_FILE).unlink()
    before = snapshot(checkout)
    with pytest.raises(SystemExit, match="is missing"):
        gen.patch(checkout)
    assert snapshot(checkout) == before


def test_a_missing_helper_source_leaves_the_checkout_unchanged(checkout: Path, monkeypatch):
    before = snapshot(checkout)
    monkeypatch.setattr(gen, "HELPER_SOURCE", checkout / "missing-helper.py")
    with pytest.raises(FileNotFoundError, match="missing-helper.py"):
        gen.patch(checkout)
    assert snapshot(checkout) == before


@pytest.mark.parametrize("stale_helper", [False, True])
def test_a_failed_helper_write_restores_the_checkout(checkout: Path, monkeypatch, stale_helper):
    scheduler = checkout / gen.SCHEDULER_FILE
    scheduler.chmod(0o751)
    original_mode = stat.S_IMODE(scheduler.stat().st_mode)
    scheduler_helper = scheduler.with_name(gen.HELPER_NAME)
    attention_helper = (checkout / gen.ATTN_FILE).with_name(gen.HELPER_NAME)
    if stale_helper:
        scheduler_helper.write_bytes(b"# previous helper\r\n")
    before = snapshot(checkout)

    replace = Path.replace

    def refuse_replace(path, destination):
        if destination == attention_helper:
            assert scheduler_helper.read_bytes() != before.get(scheduler_helper.relative_to(checkout))
            raise OSError("helper write failed")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", refuse_replace)
    with pytest.raises(OSError, match="helper write failed"):
        gen.patch(checkout)
    assert snapshot(checkout) == before
    assert stat.S_IMODE(scheduler.stat().st_mode) == original_mode


def test_a_folder_that_is_not_a_checkout_says_so(tmp_path: Path):
    with pytest.raises(SystemExit, match="No Hunyuan3D-2.1 checkout"):
        gen.patch(tmp_path / "elsewhere")


def test_check_reports_without_changing_anything(checkout: Path, capsys):
    assert gen.main(["--check", "--root", str(checkout)]) == 1
    assert "not patched" in capsys.readouterr().out
    gen.patch(checkout)
    assert gen.main(["--check", "--root", str(checkout)]) == 0
    assert "patched" in capsys.readouterr().out


@pytest.mark.posix
def test_every_patch_says_which_one_it_is_when_it_fails(checkout: Path):
    """With four patches over two files, a bare "not found" sends you hunting."""
    (checkout / gen.SCHEDULER_FILE).unlink()
    with pytest.raises(SystemExit, match=gen.SCHEDULER_FILE):
        gen.patch(checkout)