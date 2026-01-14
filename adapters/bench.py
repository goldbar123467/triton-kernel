import csv
import time
from pathlib import Path

import torch
from adapters.torch_op import flash_attention


def benchmark_attention(batch, heads, seq_len, head_dim, num_warmup=10, num_iters=100):
    """
    Benchmark Flash Attention vs PyTorch SDPA.

    Returns:
        dict with keys: seq_len, flash_ms, pytorch_ms, flash_tflops, pytorch_tflops
    """
    torch.manual_seed(42)

    q = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device='cuda', dtype=torch.float16)

    scale = 1.0 / (head_dim ** 0.5)

    # FLOPs for attention: 2 * batch * heads * seq^2 * dim (for QK^T matmul)
    # Total attention = QK^T + softmax + attn @ V
    # Approximation: 4 * batch * heads * seq^2 * dim (2 matmuls)
    flops = 4 * batch * heads * seq_len * seq_len * head_dim

    # Warmup Flash Attention
    for _ in range(num_warmup):
        _ = flash_attention(q, k, v)
    torch.cuda.synchronize()

    # Benchmark Flash Attention
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = flash_attention(q, k, v)
    torch.cuda.synchronize()
    flash_time = (time.perf_counter() - start) / num_iters
    flash_ms = flash_time * 1000
    flash_tflops = flops / flash_time / 1e12

    # Warmup PyTorch SDPA
    for _ in range(num_warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.cuda.synchronize()

    # Benchmark PyTorch SDPA
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iters
    pytorch_ms = pytorch_time * 1000
    pytorch_tflops = flops / pytorch_time / 1e12

    return {
        'seq_len': seq_len,
        'flash_ms': flash_ms,
        'pytorch_ms': pytorch_ms,
        'flash_tflops': flash_tflops,
        'pytorch_tflops': pytorch_tflops,
        'speedup': pytorch_ms / flash_ms,
    }


def run_benchmark_sweep():
    """Run benchmark across multiple sequence lengths and save results."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    # Fixed parameters
    batch = 2
    heads = 8
    head_dim = 64

    # Sweep sequence lengths
    seq_lengths = [256, 512, 1024, 2048, 4096]

    results = []
    print(f"Benchmarking Flash Attention vs PyTorch SDPA")
    print(f"Config: batch={batch}, heads={heads}, head_dim={head_dim}")
    print("-" * 80)
    print(f"{'Seq Len':>8} {'Flash (ms)':>12} {'PyTorch (ms)':>14} {'Flash TFLOPS':>14} {'PyTorch TFLOPS':>16} {'Speedup':>10}")
    print("-" * 80)

    for seq_len in seq_lengths:
        result = benchmark_attention(batch, heads, seq_len, head_dim)
        results.append(result)

        print(
            f"{result['seq_len']:8d} "
            f"{result['flash_ms']:12.3f} "
            f"{result['pytorch_ms']:14.3f} "
            f"{result['flash_tflops']:14.2f} "
            f"{result['pytorch_tflops']:16.2f} "
            f"{result['speedup']:10.2f}x"
        )

    # Save to CSV
    output_path = Path(__file__).parent / 'benchmark_results.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("-" * 80)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    run_benchmark_sweep()
