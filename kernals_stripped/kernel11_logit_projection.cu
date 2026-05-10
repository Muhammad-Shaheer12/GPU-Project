#include <cuda_runtime.h>

// Kernel 11: final logit projection (matrix-vector per batch row).
// Input: [batch x hidden], Weights: [hidden x classes], Output: [batch x classes]
__global__ void logit_projection_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weights,
                                        float* __restrict__ output,
                                        int batch,
                                        int hidden,
                                        int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * classes;
    if (idx >= total) {
        return;
    }

    int row = idx / classes;
    int cls = idx - row * classes;

    float sum = 0.0f;
    const float* in_row = input + row * hidden;
    const float* w_col = weights + cls; // column-major access via stride

    for (int k = 0; k < hidden; ++k) {
        sum += in_row[k] * w_col[k * classes];
    }

    output[idx] = sum;
}
