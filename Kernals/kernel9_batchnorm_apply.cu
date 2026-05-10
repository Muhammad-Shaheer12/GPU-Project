#include "common.h"
#include <vector>

// Kernel 9: apply batch normalization.
// y = (x - mean) / sqrt(var + eps) * gamma + beta
__global__ void batchnorm_apply_kernel(const float* __restrict__ input,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ var,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ output,
                                       int batch,
                                       int features,
                                       float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * features;
    if (idx >= total) {
        return;
    }

    int f = idx % features;
    float x = input[idx];
    float norm = (x - mean[f]) / sqrtf(var[f] + eps);
    output[idx] = norm * gamma[f] + beta[f];
}


