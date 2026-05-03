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

static void cpu_positional_encoding(const std::vector<float>& input,
                                    std::vector<float>& output,
                                    int total_tokens,
                                    int dim) {
    for (int t = 0; t < total_tokens; ++t) {
        for (int d = 0; d < dim; ++d) {
            int pair = d / 2;
            float exponent = (2.0f * pair) / static_cast<float>(dim);
            float denom = std::exp(std::log(10000.0f) * exponent);
            float angle = static_cast<float>(t) / denom;
            float pe = (d % 2 == 0) ? std::sin(angle) : std::cos(angle);
            output[t * dim + d] = input[t * dim + d] + pe;
        }
    }
}

int main() {
    // Test config aligned with your model: dim=64, seq_len=128.
    const int batch = 2;
    const int seq_len = 128;
    const int dim = 64;
    const int total_tokens = batch * seq_len;

    std::vector<float> h_input(total_tokens * dim, 0.0f);
    for (int i = 0; i < total_tokens * dim; ++i) {
        h_input[i] = 0.001f * static_cast<float>(i % 97);
    }

    std::vector<float> h_output(total_tokens * dim, 0.0f);
    std::vector<float> h_expected(total_tokens * dim, 0.0f);

    cpu_positional_encoding(h_input, h_expected, total_tokens, dim);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int total = total_tokens * dim;
    const int blocks = (total + threads - 1) / threads;

    positional_encoding_kernel<<<blocks, threads>>>(d_input, d_output, total_tokens, dim);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < h_output.size(); ++i) {
        float diff = std::fabs(h_output[i] - h_expected[i]);
        if (diff > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 3 positional encoding: PASS\n";
    } else {
        std::cout << "Kernel 3 positional encoding: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
