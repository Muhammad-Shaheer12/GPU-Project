import torch
import time
import numpy as np
import os

# Windows DLL path handling for Python 3.8+
if os.name == 'nt':
    # cuBLAS DLLs are typically shipped with PyTorch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
    
    # CUDA runtime DLLs (like cudart64_13.dll) are in the x64 subfolder
    cuda_x64_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"
    if os.path.exists(cuda_x64_path):
        os.add_dll_directory(cuda_x64_path)

try:
    import custom_cuda_ops
except ImportError as e:
    print(f"Error: {e}")
    exit(1)

def benchmark_gemm(M, K, N, iterations=100):
    print(f"\nBenchmarking GEMM: M={M}, K={K}, N={N} ({iterations} iterations)")
    
    # 1. Setup tensors
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warm up
    for _ in range(10):
        custom_cuda_ops.gemm_cublas(A, B)
        custom_cuda_ops.gemm_custom(A, B)
    torch.cuda.synchronize()

    # 2. Test cuBLAS
    start = time.time()
    for _ in range(iterations):
        res_cublas = custom_cuda_ops.gemm_cublas(A, B)
    torch.cuda.synchronize()
    time_cublas = (time.time() - start) / iterations
    
    # 3. Test Custom "Pro" Kernel
    start = time.time()
    for _ in range(iterations):
        res_custom = custom_cuda_ops.gemm_custom(A, B)
    torch.cuda.synchronize()
    time_custom = (time.time() - start) / iterations

    # 4. Verify Accuracy
    max_diff = torch.max(torch.abs(res_cublas - res_custom)).item()
    
    # 5. Calculate TFLOPS
    # GEMM floating point ops: 2 * M * N * K
    ops = 2.0 * M * N * K
    tflops_cublas = (ops / time_cublas) / 1e12
    tflops_custom = (ops / time_custom) / 1e12

    print(f"{'Kernel':<15} | {'Latency (ms)':<15} | {'TFLOPS':<10}")
    print("-" * 45)
    print(f"{'cuBLAS (TF32)':<15} | {time_cublas*1000:15.4f} | {tflops_cublas:10.4f}")
    print(f"{'Custom (Pro)':<15} | {time_custom*1000:15.4f} | {tflops_custom:10.4f}")
    print(f"\nMax Numerical Difference: {max_diff:.2e}")
    
    speedup = time_custom / time_cublas
    print(f"cuBLAS is {speedup:.2f}x faster than Custom Pro Kernel.")

if __name__ == "__main__":
    # Test cases: Small, Medium, Large, and Production Model Layers
    test_cases = [
        (128, 128, 128),
        (512, 512, 512),
        (2048, 2048, 2048),
        (2048, 64, 128),   # Production Layer 1 (FC1)
        (2048, 128, 5)     # Production Layer 2 (FC2)
    ]
    
    for M, K, N in test_cases:
        benchmark_gemm(M, K, N)
