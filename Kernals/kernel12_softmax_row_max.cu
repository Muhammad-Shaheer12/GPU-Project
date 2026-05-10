#include "common.h"
#include <vector>

// Kernel 12: compute per-row max for softmax stability.
// Input: [batch x classes], Output: [batch]
__global__ void softmax_row_max_kernel(const float* __restrict__ input,
                                       float* __restrict__ row_max,
                                       int batch,
                                       int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    // Each thread scans a strided subset of the row.
    float local_max = -FLT_MAX;
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        local_max = fmaxf(local_max, v);
    }

    float max_val = warp_reduce_max(local_max);
    if (tid == 0) {
        row_max[row] = max_val;
    }
}


