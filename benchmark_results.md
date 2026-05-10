# Benchmark Results

Date: 2026-05-10
Machine: Windows, CUDA sm_120

## Pipeline Baseline (all 15 kernels)
- CPU time (ms): 14.8814
- GPU time (ms): 1.10525
- Status: PASS (CPU vs GPU argmax match)

Notes:
- Timing measured with cudaEvent_t in Kernals/pipeline_baseline.cu.
- Input sizes: batch=128, seq_len=128, dim=64, hidden=128, classes=5.

## Pipeline Baseline (safe optimizations pass)
- CPU time (ms): 15.8361
- GPU time (ms): 0.916928
- Status: PASS (CPU vs GPU argmax match)

Notes:
- Warp-level reductions for softmax row max/sum and argmax.
- Fused bias + Leaky ReLU path in pipeline.

## Pipeline Baseline (TF32 cuBLAS GEMM)
- CPU time (ms): 14.4414
- GPU time (ms): 94.6794
- Status: PASS (CPU vs GPU argmax match)

Notes:
- cuBLAS TF32 GEMM used for hidden layer; warm-up included.
- For this small matrix size, TF32 GEMM is slower than the tiled kernel.

## Pipeline Baseline (fused softmax, tiled GEMM)
- CPU time (ms): 15.171
- GPU time (ms): 0.979456
- Status: PASS (CPU vs GPU argmax match)

Notes:
- Fused softmax kernel uses shared memory for max/sum and writes probabilities directly.

## Pipeline Baseline (fused softmax + TF32 cuBLAS GEMM)
- CPU time (ms): 14.9177
- GPU time (ms): 110.049
- Status: PASS (CPU vs GPU argmax match)

Notes:
- TF32 remains slower for this small matrix size.
