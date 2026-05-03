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

static void cpu_weighted_mean_pooling(const std::vector<float>& input,
                                      const std::vector<float>& weights,
                                      std::vector<float>& output,
                                      int batch,
                                      int seq_len,
                                      int dim) {
    for (int s = 0; s < batch; ++s) {
        for (int d = 0; d < dim; ++d) {
            float sum = 0.0f;
            float wsum = 0.0f;
            for (int t = 0; t < seq_len; ++t) {
                float w = weights[s * seq_len + t];
                sum += input[(s * seq_len + t) * dim + d] * w;
                wsum += w;
            }
            output[s * dim + d] = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
        }
    }
}

int main() {
    // Test config aligned with your model: dim=64, seq_len=128.
    const int batch = 2;
    const int seq_len = 128;
    const int dim = 64;

    std::vector<int> h_lengths = {64, 100};
    std::vector<float> h_input(batch * seq_len * dim, 0.0f);
    std::vector<float> h_weights(batch * seq_len, 0.0f);

    for (int s = 0; s < batch; ++s) {
        for (int t = 0; t < seq_len; ++t) {
            h_weights[s * seq_len + t] = (t < h_lengths[s]) ? 1.0f : 0.0f;
            for (int d = 0; d < dim; ++d) {
                h_input[(s * seq_len + t) * dim + d] = 0.01f * static_cast<float>(s * 1000 + t * 10 + d);
            }
        }
    }

    std::vector<float> h_output(batch * dim, 0.0f);
    std::vector<float> h_expected(batch * dim, 0.0f);

    cpu_weighted_mean_pooling(h_input, h_weights, h_expected, batch, seq_len, dim);

    float* d_input = nullptr;
    float* d_weights = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(seq_len, 1, 1);
    dim3 grid(batch, dim, 1);
    size_t shared_bytes = static_cast<size_t>(seq_len) * sizeof(float) * 2;

    weighted_mean_pooling_kernel<<<grid, block, shared_bytes>>>(
        d_input,
        d_weights,
        d_output,
        batch,
        seq_len,
        dim);

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
        std::cout << "Kernel 4 weighted mean pooling: PASS\n";
    } else {
        std::cout << "Kernel 4 weighted mean pooling: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
