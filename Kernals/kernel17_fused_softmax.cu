#include "common.h"

// Fused softmax: Row-wise max, sum of exps, and normalization in one kernel.
// Requires shared memory for block-level reduction if classes > warp size.
// Current implementation assumes 1 warp per row for performance.
__global__ void softmax_fused_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int batch,
                                     int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    // Shared memory for max and sum
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + 1;

    // Step 1: Find max
    float local_max = -FLT_MAX;
    for (int c = tid; c < classes; c += blockDim.x) {
        local_max = fmaxf(local_max, input[row * classes + c]);
    }
    float max_val = warp_reduce_max(local_max);
    if (tid == 0) *s_max = max_val;
    __syncthreads();
    max_val = *s_max;

    // Step 2: Sum of exps
    float local_sum = 0.0f;
    for (int c = tid; c < classes; c += blockDim.x) {
        local_sum += expf(input[row * classes + c] - max_val);
    }
    float sum_val = warp_reduce_sum(local_sum);
    if (tid == 0) *s_sum = sum_val;
    __syncthreads();
    sum_val = *s_sum;

    // Step 3: Normalize
    for (int c = tid; c < classes; c += blockDim.x) {
        int idx = row * classes + c;
        output[idx] = expf(input[idx] - max_val) / sum_val;
    }
}
