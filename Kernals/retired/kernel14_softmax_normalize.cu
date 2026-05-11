#include "common.h"
#include <vector>

// Kernel 14: normalize softmax probabilities.
// Input: logits [batch x classes], row_max [batch], row_sum [batch]
// Output: probs [batch x classes]
__global__ void softmax_normalize_kernel(const float* __restrict__ input,
                                         const float* __restrict__ row_max,
                                         const float* __restrict__ row_sum,
                                         float* __restrict__ output,
                                         int batch,
                                         int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * classes;
    if (idx >= total) {
        return;
    }

    int row = idx / classes;
    float max_val = row_max[row];
    float denom = row_sum[row];
    float v = input[idx];

    output[idx] = expf(v - max_val) / denom;
}


