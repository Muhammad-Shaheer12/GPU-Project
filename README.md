# GPU-Accelerated Sentiment Analysis Engine
---

## Project Goal
We have developed a sentiment engine using **17 CUDA kernels** to predict 1–5 star ratings from 7M Yelp reviews. By migrating the inference pipeline to a native CUDA C++ extension with a **custom register-tiled GEMM**, we achieve significant speedups and full hardware control over standard PyTorch.

---

## Development Roadmap
*   **Environment & Data**: Initialized CUDA v13.2 environment on RTX 5060; tokenized 7M Yelp reviews into a binary `.npz` format for fast GPU ingestion.
*   **Baseline Model**: Developed the `ControlledModel` in PyTorch and generated high-accuracy weights (`.pth`) to serve as the ground-truth for CUDA migration.
*   **C++ Extension**: Built the PyBind11 bridge (`custom_cuda_ops`) to allow low-latency tensor passing between the Python host and CUDA device.
*   **Kernel Development**: Implemented **17 hand-written CUDA kernels** for the full NLP pipeline (Embeddings, Pooling, BatchNorm, Softmax, etc.).
*   **High-Perf Optimizations**: Replaced standard libraries with a **Custom Tiled GEMM** and implemented **Kernel Fusion** to eliminate memory round-trips.
*   **System Refactor**: Successfully deployed a **Unity Build** system and finalized the production architecture (`custom_pipeline` logic).

---

## Folder Structure
*   **`custom_ext/`**: The C++/PyBind11 bridge and build system.
    *   `custom_ops.cpp`: PyBind11 bindings interfacing with PyTorch.
    *   `custom_kernels_wrapper.cu`: Unity Build entry point for kernel launches.
    *   `setup.py`: Build script configured for MSVC/CUDA compilation.
    *   `bridge.hpp`: Central header for host-device interface.
*   **`custom_pipeline/`**: Optimized production inference environment.
    *   `pyModel.py`: Model architecture using fused kernels and custom GEMM.
    *   `inference.py`: Production-level inference benchmarking script.
*   **`Kernals/`**: The raw CUDA kernel implementations.
    *   `kernel1-17.cu`: Individual hand-optimized source files.
    *   `common.h`: Shared CUDA utilities and reduction primitives.
    *   `retired/`: Previous iterations of kernels (superseded by fusion).
*   **`tests/`**: Automated verification and profiling suite.
    *   `run_all_tests.py`: Differential verification vs PyTorch and latency profiling.
    *   `compare_full_model.py`: End-to-end pipeline accuracy and throughput tests.
    *   `benchmark_gemm.py`: Deep-dive comparison between cuBLAS and the Custom Tiled Kernel.
*   **`scripts/`**: Offline data preparation and model training utilities.
*   **`performance_report.md`**: Technical deep-dive into benchmarks and hardware metrics.

---

## The 17 Custom Kernels
*   **K1: Pad/Truncate**: Standardizes input sequences to 128 tokens.
*   **K2: Embedding Lookup**: Vectorized retrieval of word/position features.
*   **K3: Sinusoidal PE**: Transformer-style hardware-accelerated positional encoding.
*   **K4: Weighted Pooling**: Block-level parallel reduction of sequence data into vectors.
*   **K5: Bias Add**: Parallel broadcast addition of 1D biases to 2D tensors.
*   **K6: Leaky ReLU**: Element-wise activation with configurable slope.
*   **K7: BN Mean**: Grid-stride loop calculation of feature-wise means.
*   **K8: BN Var**: Warp-level reduction of statistical variance.
*   **K9: BN Apply**: Fusion kernel for scaling, shifting, and normalization.
*   **K10: Register-Tiled GEMM**: Custom 4x4 register-tiled kernel with vectorized shared memory loads.
*   **K11: Logit Projection**: Optimized matrix-vector projection for classification.
*   **K12: Softmax Max**: Numerically stable per-row maximum finding.
*   **K13: Softmax Sum**: Parallel reduction of exponential sums.
*   **K14: Softmax Norm**: Normalization of scores into probabilities.
*   **K15: Argmax**: Warp-level reduction to find the highest-rated class.
*   **K16: Fused Bias+ReLU**: Memory-optimized combination of bias and activation.
*   **K17: Fused Softmax**: Single-pass high-efficiency softmax implementation.

---

## Benchmarking Matrix Multiplication
The project includes a specialized tool to compare our hand-optimized **Register-Tiled GEMM (K10)** against **NVIDIA cuBLAS**.

**Run the benchmark:**
```powershell
python tests/benchmark_gemm.py
```
This script sweeps across different matrix sizes (Small, Medium, Large) and reports:
- **Latency (ms)**: Total execution time.
- **TFLOPS**: Effective floating-point throughput.
- **Winner**: Identifies which implementation is faster for each workload.

---

### The Recommended Workflow
```
Write / modify a kernel
        ↓
python tests/run_all_tests.py   ← Did THIS kernel get the math right?
        ↓ PASS
Ship it
```

---

## Usage
1.  **Prepare Data**: `python scripts/prepare_data.py`
2.  **Train Model**: `python scripts/train.py`
3.  **Run Inference**: `python custom_pipeline/inference.py`
4.  **Verify Kernels**: `python tests/run_all_tests.py`
5.  **GEMM Benchmark**: `python scripts/benchmark_gemm.py`

---

## Academic Mapping: PMPP Book Chapters
| Chapter | Core Concept | Project Implementation |
| :--- | :--- | :--- |
| **Chapter 3** | Multidimensional Grids | **K2 (Embedding)** and **K10 (GEMM)** use 2D grids and blocks to process matrix data. |
| **Chapter 4** | Architecture & Scheduling | Optimized warp-level execution in **K7-9 (BatchNorm)** to maximize SM occupancy. |
| **Chapter 5** | Memory Locality | **K10 (Tiled GEMM)** utilizes Shared Memory Tiling to minimize global memory traffic. |
| **Chapter 10** | Reduction & Divergence | **K4 (Pooling)** and **K15 (Argmax)** use high-performance warp-shuffle reductions. |
| **Chapter 16** | Deep Learning | The entire project implements a complete forward inference pipeline for NLP. |
| **Chapter 19** | Computational Thinking | **K10** uses Register Tiling (4x4) to maximize arithmetic intensity per thread. |

---
