# Performance Report: Modular CUDA Pipeline

This document consolidates the benchmarking and profiling results for the RTX 5060 deployment of the modular CUDA pipeline.

## 1. Overall Pipeline Performance
*Measured via `tests/compare_full_model.py` and `Kernals/pipeline_baseline.cu`.*

| Metric | Baseline (PyTorch CUDA) | Custom CUDA Pipeline | Speedup |
| :--- | :--- | :--- | :--- |
| Forward Pass (128 samples) | ~1.10 ms | ~0.92 ms | **1.20x** |
| Memory Footprint | ~14.88 MB | ~1.10 MB | **13.5x** |

> [!NOTE]
> The custom pipeline achieves significant speedup primarily by reducing kernel launch overhead and utilizing fused kernels for operations like Bias + Leaky ReLU.

## 2. Individual Kernel Deep Dive
*Measured via `tests/profile_all_kernels.py` and NVIDIA `ncu`.*

| Kernel | Torch Benchmark | Hardware Latency (`ncu`) | Bandwidth Utilization | Occupancy |
| :--- | :--- | :--- | :--- | :--- |
| **K1: Pad/Truncate** | 30.24 µs | 12.19 µs | 55.50 GB/s (19.39%) | 74.29% |
| **K2: Embedding** | 638.10 µs | 412.15 µs | 12.10 GB/s (4.23%) | 85.12% |
| **K10: Tiled GEMM** | 47.05 µs | 28.12 µs | 145.2 GB/s (50.6%) | 92.15% |
| **K12-14: Softmax** | 77.34 µs | 34.20 µs | 89.40 GB/s (31.2%) | 65.40% |

## 3. Optimization Highlights
- **Fused Bias + ReLU**: Consolidates two kernel launches into one, reducing global memory round-trips.
- **Warp-level Reductions**: Uses `__shfl_down_sync` instead of shared memory for row-wise reductions (Max, Sum, Argmax), improving throughput for small classes.
- **Vectorized Loads**: Employs `float4` to saturate memory bandwidth in memory-bound kernels.

---

*Last Updated: 2026-05-10*
