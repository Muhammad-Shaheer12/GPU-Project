# GPU-Accelerated Sentiment Analysis Engine
---

## Project Goal
We have developed a high-performance sentiment engine using **15 custom CUDA kernels** to predict 1–5 star ratings from 7M Yelp reviews. By migrating the entire PyTorch inference pipeline to native CUDA C++, we achieve massive speedups over standard CPU/GPU implementations.

The project follows a strict **Host/Device** architecture:
*   **The Host (Python):** Manages data loading, tokenization, and coordinates the inference pipeline via a custom PyTorch C++ extension.
*   **The Device (CUDA):** Executes all mathematical operations, from embedding lookups to the final softmax and argmax predictions, using hand-optimized CUDA kernels.

---

## Progress
1.  **Environment Setup:** Configured Python, PyTorch, and NVIDIA CUDA Toolkit (v13.2).
2.  **Data Acquisition:** Processed the Yelp Academic Dataset (Customer Reviews).
3.  **Data Pipeline:** Tokenized 7 million reviews into `yelp_tokenized.npz`.
4.  **Baseline Training:** Trained the model (`pyModel.py`) to generate weights (`controlled_model_weights.pth`).
5.  **Kernel Migration:** Unified all 15 kernels into a single library-style header/source structure, enabling conditional compilation via `#ifndef PIPELINE_BUILD` for flexible testing and integration.
6.  **Extension Development:** Built a PyBind11 C++ extension (`custom_cuda_ops`) to bridge Python and CUDA.
7.  **Full Integration:** Replaced all forward pass operations in the model with custom CUDA kernel calls.
8.  **Verification:** Validated the end-to-end pipeline with real review data, achieving identical accuracy with significant performance gains.

---

## The 15 Custom Kernels

### Data Preparation
1.  **Token Padding & Truncation**: Standardizes sequence lengths with a 1-thread-per-token mapping.
2.  **Vectorized Embedding Lookup**: Uses `float4` vectorized reads to maximize memory bandwidth during word and position embedding retrieval.
3.  **Sinusoidal Positional Encoding**: Applies Transformer-based encoding using hardware-accelerated math functions.
4.  **Weighted Mean Pooling**: Executes block-level parallel reductions through binary tree summation in shared memory.

### Neural Layers
5.  **Bias Addition**: Broadcasts 1D bias vectors across 2D tensors using optimized column indexing.
6.  **Leaky ReLU Activation**: Element-wise non-linearity with configurable alpha.
7.  **BatchNorm Mean**: Calculates feature means via grid-stride loops and warp-level reductions.
8.  **BatchNorm Variance**: Computes statistical variance using register-to-register communication.
9.  **BatchNorm Apply**: Normalizes, scales, and shifts data in a single fusion kernel.
10. **Tiled GEMM**: High-performance matrix multiplication ($C = A \times B$) using shared memory tiling.
11. **Logit Projection**: Optimized matrix-vector projection for the final classification layer.

### Classification Pipeline
12. **Softmax Row Max**: Computes per-row maximums for numerical stability.
13. **Softmax Row Sum**: Calculates the sum of exponentials ($e^{x - max}$) using shared memory reduction.
14. **Softmax Normalize**: Produces final probabilities by normalizing against the row sum.
15. **Argmax**: Determines the final 1-5 star prediction by finding the index of the maximum probability.

---

## File Structure
```text
GPUProject/
├── custom_ext/               # PyTorch C++ Extension
│   ├── custom_ops.cpp        # PyBind11 bindings
│   ├── custom_kernels_wrapper.cu # CUDA launch wrappers
│   └── setup.py              # Build configuration
├── custom_pipeline/          # Integrated Model Logic
│   ├── pyModel.py            # Model with custom forward pass
│   └── inference.py          # End-to-end verification script
├── Kernals/                  # Source kernels & Shared Utilities
│   ├── common.h              # Shared CUDA macros & primitives
│   └── kernelX_...cu         # Individual library-ready kernels
├── tests/                    # Benchmarking & Validation
├── profiling_results.md      # Detailed Performance Deep Dive
├── scripts/                  # Preprocessing and baseline scripts
├── weights/                  # Trained model weights (.pth)
├── dataset/                  # Raw Yelp JSON data
└── README.md
```

---

## How to Run

### 1. Build the Custom CUDA Extension
Compile the kernels and register them with PyTorch:
```powershell
cd custom_ext
python setup.py install
```

### 2. Run the Full Custom Inference
Execute the end-to-end sentiment analysis using all 15 kernels:
```powershell
python custom_pipeline/inference.py
```

### 3. (Optional) Re-train or Pre-process
To regenerate weights or tokenized data:
```powershell
python scripts/preprocessing.py
python custom_pipeline/pyModel.py  # Runs training mode
```

---

**Current Status:** All 15 kernels are fully integrated and verified. The system is capable of performing high-speed sentiment inference on millions of reviews using native GPU acceleration.
