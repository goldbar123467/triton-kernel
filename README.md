# Flash Attention Triton Kernel

A memory-efficient Flash Attention implementation in Triton with online softmax algorithm.

## Features

- Online softmax for O(N) memory complexity vs O(N^2) in standard attention
- Configurable block sizes (BLOCK_M=64, BLOCK_N=64)
- FP32 accumulator for numerical stability, FP16 output
- Supports batch, heads, seq_len, head_dim layout
- Drop-in replacement for PyTorch SDPA

## Installation

```bash
pip install torch triton pytest
```

## Usage

```python
from attention import flash_attention
import torch

# Input: (batch, heads, seq_len, head_dim)
q = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device='cuda')

# Run flash attention
output = flash_attention(q, k, v)  # (2, 8, 1024, 64)
```

## How It Works

Flash Attention uses an online softmax algorithm to achieve memory efficiency:

- **Tiled computation**: Processes attention in blocks rather than materializing full attention matrix
- **Online statistics**: Maintains running max and sum for numerically stable softmax without storing intermediate results
- **Fused operations**: Combines softmax and matrix multiplication in a single kernel pass
- **Memory complexity**: Reduces memory from O(N^2) to O(N) for sequence length N

## Testing

```bash
pytest test_attn.py -v
```

All 5 tests pass in approximately 2.5 seconds.

## Benchmarking

```bash
python -m adapters.bench
```

Compares performance against PyTorch's `scaled_dot_product_attention`.

## Project Structure

```
attention/
├── __init__.py          # Public API exports
├── core/flash_attn.py   # Triton kernel implementation
├── adapters/torch_op.py # PyTorch wrapper
├── adapters/bench.py    # Benchmark suite
└── test_attn.py         # Test suite
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+
- CUDA-capable GPU

## License

MIT
