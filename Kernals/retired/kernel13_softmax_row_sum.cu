#include "common.h"
#include <vector>

// Kernel 13: compute per-row sum of exp(x - row_max).
// Input: logits [batch x classes], row_max [batch], Output: row_sum [batch]
__global__ void softmax_row_sum_kernel(const float* __restrict__ input,
                                       const float* __restrict__ row_max,
                                       float* __restrict__ row_sum,
                                       int batch,
                                       int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    float max_val = row_max[row];
    float local_sum = 0.0f;
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        local_sum += expf(v - max_val);
    }

    float sum_val = warp_reduce_sum(local_sum);
    if (tid == 0) {
        row_sum[row] = sum_val;
    }
}


