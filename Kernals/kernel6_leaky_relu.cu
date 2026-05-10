#include "common.h"
#include <vector>

// Kernel 6: Leaky ReLU activation (y = x if x >= 0, else alpha * x).
__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int total,
                                  float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    float x = input[idx];
    output[idx] = (x >= 0.0f) ? x : (alpha * x);
}


