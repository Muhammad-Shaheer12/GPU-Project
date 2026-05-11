# GPU-Accelerated Sentiment Analysis Engine
---

## Project Goal
We have developed a high-performance sentiment engine using **17 hand-optimized CUDA kernels** to predict 1–5 star ratings from 7M Yelp reviews. By migrating the inference pipeline to a native CUDA C++ extension with **cuBLAS** acceleration, we achieve significant speedups over standard PyTorch.

---

## Development Roadmap
*   **[Phase 1] Environment & Data**: Initialized CUDA v13.2 environment on RTX 5060; tokenized 7M Yelp reviews into a binary `.npz` format for fast GPU ingestion.
*   **[Phase 2] Baseline Model**: Developed the `ControlledModel` in PyTorch and generated high-accuracy weights (`.pth`) to serve as the ground-truth for CUDA migration.
*   **[Phase 3] C++ Extension**: Built the PyBind11 bridge (`custom_cuda_ops`) to allow low-latency tensor passing between the Python host and CUDA device.
*   **[Phase 4] Kernel Development**: Implemented 15 hand-written CUDA kernels for the full NLP pipeline (Embeddings, Pooling, BatchNorm, Softmax, etc.).
*   **[Phase 5] High-Perf Optimizations**: Integrated **cuBLAS** for Tensor Core acceleration and implemented **Kernel Fusion** to reduce memory round-trips.
*   **[Phase 6] System Refactor**: Deployed a **Unity Build** system and refactored the project into a clean production structure (`custom_pipeline` vs `scripts`).

---

## Folder Structure
*   **`custom_ext/`**: The C++/CUDA extension source.
    *   `custom_ops.cpp`: PyBind11 bindings for PyTorch.
    *   `custom_kernels_wrapper.cu`: Unity Build launch pad for all kernels.
    *   `bridge.hpp`: Centralized C++/CUDA interface.
*   **`custom_pipeline/`**: Optimized inference environment.
    *   `pyModel.py`: Final architecture (Pool -> FusedReLU -> BN -> cuBLAS GEMM).
    *   `inference.py`: Production inference benchmark.
*   **`scripts/`**: Data preparation and training.
    *   `prepare_data.py`: Merged tokenization and vocabulary builder.
    *   `train.py`: Model training and weights generation.
*   **`Kernals/`**: The raw CUDA kernel implementations.
*   **`tests/`**: Automated verification and profiling suite.

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

## Performance & Optimization
Detailed benchmarks, individual kernel latencies, and technical optimization deep-dives can be found in the [Performance Report](performance_report.md).

---

## Testing Strategy
The project uses a Python-based unit test suite to validate every kernel individually against PyTorch's native math.

### `tests/run_all_tests.py` — Per-Kernel Unit Tests
Tests each of the 17 kernels **individually and in isolation**. If a kernel produces wrong numbers, you see exactly which one failed.
```
K1: Pad/Truncate     → Verifying... PASS   120.45 us
K2: Embedding Lookup → Verifying... PASS   340.12 us
K7: BN Mean          → Verifying... FAIL   ← precise, you know exactly what broke
```

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

**Current Status:** Production ready. All 17 kernels are verified and optimized for RTX 50-series hardware.
