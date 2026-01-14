import pytest
import torch
from adapters.torch_op import flash_attention


def ref_attention(q, k, v):
    """Standard attention for reference."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn = torch.softmax(q.float() @ k.float().transpose(-2, -1) * scale, dim=-1)
    return (attn @ v.float()).to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch,heads,seq_len,head_dim", [
    (2, 4, 64, 32),
    (2, 8, 256, 64),
    (1, 16, 512, 64),
])
def test_flash_attention_correctness(batch, heads, seq_len, head_dim):
    torch.manual_seed(42)
    q = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)

    out = flash_attention(q, k, v)
    ref = ref_attention(q, k, v)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_shape_and_dtype():
    batch, heads, seq_len, head_dim = 2, 4, 128, 64
    q = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)

    out = flash_attention(q, k, v)
    assert out.shape == (batch, heads, seq_len, head_dim)
    assert out.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_deterministic():
    torch.manual_seed(42)
    batch, heads, seq_len, head_dim = 2, 4, 128, 64
    q = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)

    out1 = flash_attention(q, k, v)
    out2 = flash_attention(q, k, v)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
