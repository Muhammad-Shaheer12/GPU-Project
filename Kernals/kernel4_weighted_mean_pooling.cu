#include "common.h"
#include <vector>

// Kernel 4: weighted mean pooling across sequence length.
// One block handles one (sentence, dim) pair and reduces over seq_len.
__global__ void weighted_mean_pooling_kernel(const float* __restrict__ input,
                                             const float* __restrict__ weights,
                                             float* __restrict__ output,
                                             int batch,
                                             int seq_len,
                                             int dim) {
    int sentence = blockIdx.x;
    int d = blockIdx.y;
    int t = threadIdx.x; // token index within the sentence

    if (sentence >= batch || d >= dim || t >= seq_len) {
        return;
    }

    int token_idx = sentence * seq_len + t;
    float w = weights[token_idx];
    float v = input[token_idx * dim + d];

    extern __shared__ float shared[];
    float* shared_val = shared;                  // seq_len floats
    float* shared_w = shared + seq_len;          // seq_len floats

    shared_val[t] = v * w;
    shared_w[t] = w;
    __syncthreads();

    // Parallel reduction over seq_len.
    for (int stride = seq_len / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            shared_val[t] += shared_val[t + stride];
            shared_w[t] += shared_w[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        float denom = shared_w[0];
        output[sentence * dim + d] = (denom > 0.0f) ? (shared_val[0] / denom) : 0.0f;
    }
}


