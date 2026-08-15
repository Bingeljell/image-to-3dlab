"""Static contract for the cheap MPS attention integration gate."""

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_trellis_space_attention.py"


def test_gate_covers_stage3_cross_attention_and_sparse_self_attention():
    source = SCRIPT.read_text()
    assert '"self":' in source
    assert '"cross":' in source
    assert "sparse_scaled_dot_product_attention" in source
    assert 'choices=("sdpa", "metal_flash")' in source
    assert 'channels = 128 if args.backend == "sdpa" else 64' in source
    assert "torch.allclose" in source
