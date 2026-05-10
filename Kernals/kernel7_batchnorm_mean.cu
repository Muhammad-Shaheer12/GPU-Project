#include "common.h"
#include <vector>

// Kernel 7: compute mean per feature across the batch.
// Input shape: [batch, features], row-major.
__global__ void batchnorm_mean_kernel(const float* __restrict__ input,
                                      float* __restrict__ mean,
                                      int batch,
                                      int features) {
    int feature = blockIdx.x;
    int tid = threadIdx.x;

    if (feature >= features) {
        return;
    }

    float sum = 0.0f;
    for (int i = tid; i < batch; i += blockDim.x) {
        sum += input[i * features + feature];
    }

    // Reduce within each warp.
    float warp_sum = warp_reduce_sum(sum);

    // One value per warp goes to shared memory.
    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction by the first warp.
    float block_sum = 0.0f;
    if (warp_id == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            mean[feature] = block_sum / static_cast<float>(batch);
        }
    }
}


