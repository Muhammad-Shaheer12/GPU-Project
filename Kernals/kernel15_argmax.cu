#include "common.h"
#include <vector>

// Warp-level argmax using shuffle.
__device__ void warp_reduce_argmax(float& v, int& idx) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_v = __shfl_down_sync(0xFFFFFFFF, v, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_v > v) {
            v = other_v;
            idx = other_idx;
        }
    }
}

// Kernel 15: argmax per row.
// Input: probs [batch x classes], Output: argmax indices [batch]
__global__ void argmax_kernel(const float* __restrict__ input,
                              int* __restrict__ output,
                              int batch,
                              int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        if (v > local_max) {
            local_max = v;
            local_idx = c;
        }
    }

    warp_reduce_argmax(local_max, local_idx);
    if (tid == 0) {
        output[row] = local_idx;
    }
}


