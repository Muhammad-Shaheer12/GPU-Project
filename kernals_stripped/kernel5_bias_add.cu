#include <cuda_runtime.h>

// Kernel 5: add a bias vector to each row of a 2D tensor.
__global__ void bias_add_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int rows,
                                int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }

    int col = idx % cols;
    output[idx] = input[idx] + bias[col];
}
