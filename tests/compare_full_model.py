import torch
import time
import numpy as np
import os
import sys

# Ensure we can import ControlledModel
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_pipeline"))
from pyModel import ControlledModel

def benchmark_full_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available. Benchmarking on CPU is not recommended for this test.")
        return

    # Setup model parameters
    vocab_size = 50000
    batch_size = 2048
    seq_len = 128
    
    model = ControlledModel(vocab_size=vocab_size).to(device)
    model.eval()
    
    # Create sample inputs
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    
    print(f"Benchmarking Full Model Forward Pass (Batch Size: {batch_size}, Seq Len: {seq_len})")
    print("-" * 60)

    # 1. Warm-up
    print("Warming up...")
    for _ in range(10):
        _ = model(inputs, lengths=lengths)
    torch.cuda.synchronize()

    # 2. Benchmark Custom CUDA Implementation
    print("Measuring Custom CUDA Pipeline...")
    start_time = time.time()
    iters = 100
    for _ in range(iters):
        _ = model(inputs, lengths=lengths)
    torch.cuda.synchronize()
    custom_time = (time.time() - start_time) / iters * 1000 # in ms

    # 3. Benchmark Standard PyTorch Implementation
    # We force the fallback by temporarily disabling the custom ops check in the model
    print("Measuring Standard PyTorch Implementation...")
    
    # To force fallback, we'll monkeypatch the custom_cuda_ops import check in the instance
    import pyModel
    orig_custom = pyModel.custom_cuda_ops
    pyModel.custom_cuda_ops = None # Force fallback
    
    start_time = time.time()
    for _ in range(iters):
        _ = model(inputs, lengths=lengths)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / iters * 1000 # in ms
    
    # Restore original custom ops for future use
    pyModel.custom_cuda_ops = orig_custom

    # 4. Results
    print("-" * 60)
    print(f"Standard PyTorch (CUDA): {pytorch_time:.4f} ms")
    print(f"Custom CUDA Kernels:     {custom_time:.4f} ms")
    print(f"Speedup:                 {pytorch_time / custom_time:.2f}x")
    print("-" * 60)

if __name__ == "__main__":
    benchmark_full_model()
