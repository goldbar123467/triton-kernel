import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    num_heads, seq_len, head_dim, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(seq_len, BLOCK_M)

    block_m_idx = pid % num_blocks_m
    head_idx = (pid // num_blocks_m) % num_heads
    batch_idx = pid // (num_blocks_m * num_heads)

    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # load Q block [BLOCK_M, BLOCK_D]
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)

    # online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    num_blocks_n = tl.cdiv(seq_len, BLOCK_N)
    for block_n_idx in range(num_blocks_n):
        offs_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        # load K block [BLOCK_N, BLOCK_D]
        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)

        # QK^T [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= scale

        # mask out-of-bounds
        qk = tl.where((offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len), qk, float("-inf"))

        # online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # load V block [BLOCK_N, BLOCK_D]
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)

        # update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    # normalize
    acc = acc / l_i[:, None]

    # store output
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim))


def flash_attention_forward(q, k, v):
    batch, heads, seq_len, head_dim = q.shape
    assert k.shape == v.shape == q.shape

    o = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    scale = 1.0 / (head_dim ** 0.5)
    grid = (batch * heads * triton.cdiv(seq_len, BLOCK_M),)

    flash_attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        heads, seq_len, head_dim, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return o
