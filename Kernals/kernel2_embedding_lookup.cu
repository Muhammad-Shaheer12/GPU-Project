#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Minimal CUDA error checking for fast feedback during kernel development.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

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

static void cpu_embedding_lookup(const std::vector<int>& tokens,
                                 const std::vector<float>& embedding,
                                 std::vector<float>& output,
                                 int total_tokens,
                                 int dim,
                                 int vocab,
                                 int unk_id) {
    for (int t = 0; t < total_tokens; ++t) {
        int token = tokens[t];
        if (token < 0 || token >= vocab) {
            token = unk_id;
        }
        int emb_offset = token * dim;
        int out_offset = t * dim;
        for (int d = 0; d < dim; ++d) {
            output[out_offset + d] = embedding[emb_offset + d];
        }
    }
}

int main() {
    // Test config aligned with your model: dim=64, seq_len=128.
    const int batch = 2;
    const int seq_len = 128;
    const int dim = 64;
    const int vocab = 103569;
    const int unk_id = 1;

    const int total_tokens = batch * seq_len;

    std::vector<int> h_tokens(total_tokens, 0);
    for (int i = 0; i < total_tokens; ++i) {
        h_tokens[i] = (i * 37) % (vocab + 5); // inject a few out-of-range tokens
    }

    std::vector<float> h_embedding(vocab * dim, 0.0f);
    for (int v = 0; v < vocab; ++v) {
        for (int d = 0; d < dim; ++d) {
            h_embedding[v * dim + d] = 0.01f * static_cast<float>(v) + static_cast<float>(d);
        }
    }

    std::vector<float> h_output(total_tokens * dim, 0.0f);
    std::vector<float> h_expected(total_tokens * dim, 0.0f);

    cpu_embedding_lookup(h_tokens, h_embedding, h_expected, total_tokens, dim, vocab, unk_id);

    int* d_tokens = nullptr;
    float* d_embedding = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tokens, h_tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_embedding, h_embedding.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), h_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_embedding, h_embedding.data(), h_embedding.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 8, 1); // 16 lanes * 4 dims = 64 dims; 8 tokens per block in Y.
    dim3 grid(1, (total_tokens + block.y - 1) / block.y, 1);

    embedding_lookup_kernel<<<grid, block>>>(
        d_tokens,
        d_embedding,
        d_output,
        total_tokens,
        dim,
        vocab,
        unk_id);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < h_output.size(); ++i) {
        float diff = std::fabs(h_output[i] - h_expected[i]);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 2 embedding lookup: PASS\n";
    } else {
        std::cout << "Kernel 2 embedding lookup: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_embedding));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
