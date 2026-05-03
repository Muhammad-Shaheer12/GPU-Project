# GPU-Accelerated Sentiment Analysis Engine
---

## Project goal
We are developing a high-performance sentiment engine using custom CUDA kernels to predict 1–5 star ratings from 7M Yelp reviews.

The project follows a strict **Host/Device** architecture to maximize hardware efficiency:
*   **The Host (Python):** It cleans text, tokenizing words, and manages dataflow.
*   **The Device (CUDA):** All heavy mathematical operations—including 2D embedding lookups, matrix multiplications (GEMM), and the final softmax prediction—are handled by 15 custom kernels written from scratch in CUDA C++.

---

## Progress
1.  **Environment Setup:** We first set up the libraries and the software required for using Python with CUDA and C so that they don't mess up later on.
2.  **Data Acquisition:** We downloaded the Yelp Academic Dataset (Customer Reviews) and extracted the raw JSON files. (link: https://business.yelp.com/data/resources/open-dataset/)
3.  **Data Pipeline:** We unzipped the data and ran our custom scripts for data loading and tokenization. This produced `yelp_tokenized.npz`, a dense binary file of tokens for the model.
4.  **Model Construction:** We implemented the actual ML model (`pyModel.py`) in python to establish our 5-point rating logic.
5.  **Baseline Training:** We trained the model on 7 million reviews to generate the weights and debug tensors.
6.  **Inference:** We developed a small inference file to verify our saved model weights.

---

## Kernals

i. Token Padding & Truncation: Implements a flat 1D memory layout and 1-thread-per-token mapping to handle sequence lengths while ensuring contiguous memory access.

ii. Vectorized Embedding Lookup: Utilizes a 2D thread grid and float4 vectorized reads to maximize global memory bandwidth with built-in out-of-vocabulary handling.

iii. Sinusoidal Positional Encoding: Applies Transformer-based positional formulas via residual addition using hardware-accelerated logarithmic identities for optimized computation.

iv. Weighted Mean Pooling: Executes block-level parallel reductions through binary tree summation to calculate weighted averages within partitioned shared memory.

v. Bias Addition (Broadcasting): Dynamically broadcasts a 1D bias vector across a 2D tensor using modulo-based column indexing to eliminate memory duplication.

vi. Leaky ReLU Activation: Provides element-wise non-linearity using a configurable alpha multiplier to preserve small gradients and prevent dead neurons.

vii. Batch Normalization Mean: Calculates feature column means via grid-stride loops and warp-level register reductions for high-speed computation.

viii. Batch Normalization Variance: Computes statistical variance using a warp-reduce-sum architecture to maintain maximum register-to-register communication efficiency.

Current Status: All Host-side logic is done. We have successfully tokenized the dataset, trained our baseline model, and extracted the debug tensors.

---

## File Structure
The project is organized into modular directories:
```text
    GPUProject/
    ├── dataset/                      # Yelp dataset
    ├── Kernals/                      # all the kernals that we wrote
    ├── scripts/                      # Host files
    │   ├── dataloader.py             # Batching and memory management
    │   ├── preprocessing.py          # Entry point for the tokenization pipeline
    │   ├── pyModel.py                # The PyTorch model
    │   ├── tokenizer.py              # Text cleaning and word-to-ID mapping
    │   ├── inference.py              # Inference engine using saved weights
    │   ├── yelp_tokenized.npz        # Compressed binary dataset
    │   └── cuda_debug_tensors.npz    # Intermediate tensors for CUDA verification
    ├── weights/
    │   └── controlled_model_weights.pth # Trained model weights
    ├── README.md
    └── .gitignore
```
How to run:
1. Pre-processing & Tokenization
Convert 7 million reviews into a dense binary format.
```python scripts/preprocessing.py```

2. Train the python Model
Train the PyTorch model to generate the weights and the debug benchmarks.
```python scripts/pyModel.py```

4. Run Inference
Verify that the model logic and weights are working correctly on sample data.
```python scripts/inference.py```
