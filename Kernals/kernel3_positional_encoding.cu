#include "common.h"
#include <vector>

// Kernel 3: add classic transformer positional encoding to embeddings.
__global__ void positional_encoding_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int total_tokens,
                                           int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * dim;
    if (idx >= total) {
        return;
    }

    int token = idx / dim;
    int d = idx - token * dim;

    int pair = d / 2; // shared frequency for (2i, 2i+1)
    float exponent = (2.0f * pair) / static_cast<float>(dim);
    float denom = expf(logf(10000.0f) * exponent);
    float angle = static_cast<float>(token) / denom;

    float pe = (d % 2 == 0) ? sinf(angle) : cosf(angle);
    output[idx] = input[idx] + pe;
}


