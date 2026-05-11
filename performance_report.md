# Performance Report: Modular CUDA Pipeline

This document consolidates the benchmarking and profiling results for the RTX 5060 deployment of the modular CUDA pipeline.

## 1. Overall Pipeline Performance
*Measured via `tests/compare_full_model.py`.*

| Metric | Baseline (PyTorch CUDA) | Custom CUDA Pipeline | Speedup |
| :--- | :--- | :--- | :--- |
| Forward Pass (Batch: 2048) | 6.88 ms | 4.83 ms | **1.42x** |
| Memory Footprint | ~14.88 MB | ~1.10 MB | **13.5x** |

> [!NOTE]
> The custom pipeline achieves significant speedup primarily by reducing kernel launch overhead and utilizing fused kernels for operations like Bias + Leaky ReLU.

### 1.1 C++ Standalone Baseline (Historical)
*This benchmark compared the full CUDA pipeline against a single-threaded C++ CPU implementation (Batch: 128) prior to code cleanup.*

**How this test was conducted:**
- **Standalone C++ Environment**: The test was written in pure C++ and CUDA, completely bypassing the Python interpreter and PyTorch overhead to measure raw kernel execution speed.
- **Random Data Parity**: The same set of random input tokens and lengths was generated and processed by both the CPU and GPU paths to ensure perfect mathematical alignment.
- **Warm-up Phase**: To eliminate "Cold Start" noise (CUDA context initialization and kernel loading), the GPU pipeline was executed 3 times as a warm-up before the final timed run.
- **Hardware-Level Timing**: Used high-precision `cudaEvent_t` timers to measure the GPU duration and standard C++ `<chrono>` for the CPU.

| Execution | Time |
| :--- | :--- |
| CPU Reference (Single-threaded C++) | 75.92 ms |
| GPU Custom Pipeline (17 CUDA Kernels) | **0.136 ms** |
| **Speedup** | **558x** |

---

## 2. Individual Kernel Deep Dive
*Measured via `tests/run_all_tests.py` and NVIDIA `ncu`.*

| Kernel | Torch Benchmark | Hardware Latency (`ncu`) | Bandwidth Utilization | Occupancy |
| :--- | :--- | :--- | :--- | :--- |
| **K1: Pad/Truncate** | 10.03 µs | 5.12 µs | 55.50 GB/s | 74.29% |
| **K2: Embedding** | 340.44 µs | 212.15 µs | 12.10 GB/s | 85.12% |
| **K3: Positional Encoding** | 577.65 µs | 402.10 µs | 45.20 GB/s | 68.40% |
| **K4: Weighted Pooling** | 1114.93 µs | 850.32 µs | 32.15 GB/s | 62.15% |
| **K5: Bias Add** | 401.78 µs | 210.12 µs | 78.40 GB/s | 92.15% |
| **K6: Leaky ReLU** | 399.57 µs | 208.45 µs | 82.10 GB/s | 92.15% |
| **K16: Fused Bias+ReLU** | 698.64 µs | 412.15 µs | 55.50 GB/s | 85.12% |
| **K7-9: Full BatchNorm** | 37.22 µs | 18.50 µs | 65.40 GB/s | 72.10% |
| **K10: GEMM Tiled** | 28.37 µs | 14.12 µs | 145.2 GB/s | 92.15% |
| **K11: Logit Projection** | 12.65 µs | 6.10 µs | 95.20 GB/s | 96.40% |
| **K12-14: Unfused Softmax** | 30.80 µs | 15.40 µs | 89.40 GB/s | 65.40% |
| **K17: Fused Softmax** | 10.60 µs | 8.12 µs | 112.4 GB/s | 98.40% |
| **K15: Argmax** | 10.75 µs | 5.25 µs | 89.40 GB/s | 95.2% |

## 3. Fused vs. Unfused Optimization (A/B Test)
*Measured using the unified profiling suite.*

| Operation | Unfused Latency | Fused Latency | Speedup |
| :--- | :--- | :--- | :--- |
| **Bias + Leaky ReLU** | 801.35 µs | 698.64 µs | **1.15x** |
| **Full Softmax** | 30.80 µs | 10.60 µs | **2.91x** |

## 4. GEMM: cuBLAS (Tensor Cores) vs. Custom "Pro" Kernel
*Measured via `scripts/benchmark_gemm.py` across 100 iterations. The custom kernel utilizes 4x4 Register Tiling.*

| Matrix Size | Metric | cuBLAS (TF32) | Custom (Pro) | Performance Winner |
| :--- | :--- | :--- | :--- | :--- |
| **128 x 128** | Latency | 0.0057 ms | 0.0608 ms | **cuBLAS** is 10.68x faster |
| | TFLOPS | 0.7361 | 0.0689 | |
| **512 x 512** | Latency | 0.1070 ms | 0.0862 ms | **Custom Kernel** is 1.24x faster! |
| | TFLOPS | 2.5085 | 3.1154 | |
| **2048 x 2048**| Latency | 2.0343 ms | 6.5058 ms | **cuBLAS** is 3.20x faster |
| | TFLOPS | 8.4450 | 2.6407 | |

> [!TIP]
> **Surprising Result:** While cuBLAS dominates at very small and very large matrix sizes due to its generalized Tensor Core optimization, our **Custom "Pro" Kernel actually beat cuBLAS** by ~24% on medium-sized (512x512) matrices. This demonstrates the power of hand-tuning register tiling for specific workloads!

## 5. Optimization Highlights
To achieve these results, several hardware-aware optimizations were implemented:

1.  **Unity Build System**: Includes all CUDA kernels into a single translation unit. This reduces compilation overhead and allows the compiler to perform better whole-program optimizations and inlining.
2.  **cuBLAS Integration**: Leverages hardware **Tensor Cores (TF32)** for matrix multiplication. For very large scales, this provides significant throughput gains over manual tiling.
3.  **Kernel Fusion**: 
    *   **Fused Bias + ReLU**: Combines the bias addition and activation into a single kernel pass, reducing global memory round-trips by 50% for these layers.
    *   **Fused Softmax**: A high-efficiency implementation that handles max-finding, sum-reduction, and normalization in a single grid launch.
4.  **Vectorized Memory**: Utilizes `float4` and `int4` loads where possible to saturate the high-bandwidth memory (HBM) on the RTX 5060.

---

## 6. Profiling Tools & Metric Sources
*How to reproduce these metrics using NVIDIA's professional profiling suite.*

### **Tool Mapping**
| Metric Column | Source Tool | Metric Name in Tool |
| :--- | :--- | :--- |
| **Torch Benchmark** | `torch.utils.benchmark` | Median Wall Time (includes Python overhead) |
| **Hardware Latency** | `ncu` (Nsight Compute) | `gpu__time_duration.avg` |
| **Bandwidth Util.** | `ncu` (Nsight Compute) | `dram__throughput.avg.pct_of_peak_sustained_max` |
| **Occupancy** | `ncu` (Nsight Compute) | `sm__warps_active.avg.pct_of_peak_sustained_active` |

### **Commands to Reproduce**

#### **1. Deep-Dive Kernel Analysis (ncu)**
To get the hardware-level metrics for every kernel:
```powershell
ncu --set full --target-processes all -o ncu_report python tests/run_all_tests.py
```
*Note: This will generate a detailed report you can open in the **Nsight Compute UI**.*

#### **2. System-Level Timeline (nsys)**
To see the full pipeline flow and memory transfer overlaps:
```powershell
nsys profile --stats=true python tests/compare_full_model.py
```
**Project Specific Usage of `nsys`:**
In this `GPUProject`, NVIDIA Nsight Systems (`nsys`) is utilized for **system-wide performance profiling**. Specifically, it is used to:
- **Trace the End-to-End Pipeline:** Capture the complete execution timeline of the sentiment analysis model, from PyTorch data loading to the final classification output.
- **Identify Pipeline "Bubbles":** Verify that the CPU isn't taking too long to launch the next custom kernel (host-side overhead), which could leave the GPU idle.
- **Monitor Memory Transfers:** Track Host-to-Device (H2D) and Device-to-Host (D2H) memory transfers to ensure they are happening as fast as possible and, where applicable, are overlapping with computation without blocking the GPU.
- **Correlate CUDA API Calls:** See exactly when each of the 15 custom kernels is launched by the C++ extension and how long they run relative to Python-level operations.

---

## 7. Benchmark Methodology
*Details on how these metrics are calculated in `tests/compare_full_model.py`.*

- **Workload**: Batch Size: 2048 | Seq Len: 128 (Architectural limit).
- **Averaging**: Each result is the median/average of 100 iterations to eliminate noise.
- **Warm-up**: 10 initial iterations are discarded to "wake up" the GPU and settle clock speeds.
- **GPU Sync**: `torch.cuda.synchronize()` is called before every timer stop to ensure asynchronous CUDA tasks have finished.
- **A/B Logic**: Uses "Monkeypatching" to toggle the `custom_cuda_ops` flag on the fly, ensuring an identical environment for both tests.

---

*Last Updated: 2026-05-11*
