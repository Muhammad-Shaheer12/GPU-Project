#include "common.h"
#include <vector>

// Kernel 8: compute variance per feature across the batch.
// Input shape: [batch, features], row-major.
// Mean is precomputed by Kernel 7.
__global__ void batchnorm_var_kernel(const float* __restrict__ input,
                                     const float* __restrict__ mean,
                                     float* __restrict__ var,
                                     int batch,
                                     int features) {
    int feature = blockIdx.x;
    int tid = threadIdx.x;

    if (feature >= features) {
        return;
    }

    float sum = 0.0f;
    float m = mean[feature];
    for (int i = tid; i < batch; i += blockDim.x) {
        float diff = input[i * features + feature] - m;
        sum += diff * diff;
    }

    float warp_sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            var[feature] = block_sum / static_cast<float>(batch);
        }
    }
}


