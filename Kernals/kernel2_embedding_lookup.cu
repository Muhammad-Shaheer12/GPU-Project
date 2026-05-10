#include "common.h"
#include <vector>

// Kernel 2: embedding lookup using vectorized float4 loads for coalescing.
__global__ void embedding_lookup_kernel(const int* __restrict__ tokens,
                                        const float* __restrict__ embedding,
                                        float* __restrict__ output,
                                        int total_tokens,
                                        int dim,
                                        int vocab,
                                        int unk_id) {
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane = threadIdx.x; // each lane handles 4 dims
    int base_dim = lane * 4;

    if (token_idx >= total_tokens || base_dim >= dim) {
        return;
    }

    int token = tokens[token_idx];
    if (token < 0 || token >= vocab) {
        token = unk_id;
    }

    int emb_offset = token * dim + base_dim;
    int out_offset = token_idx * dim + base_dim;

    if (base_dim + 3 < dim) {
        float4 v = reinterpret_cast<const float4*>(embedding + emb_offset)[0];
        reinterpret_cast<float4*>(output + out_offset)[0] = v;
    } else {
        // Tail case if dim is not a multiple of 4.
        for (int k = 0; k < 4 && base_dim + k < dim; ++k) {
            output[out_offset + k] = embedding[emb_offset + k];
        }
    }
}


