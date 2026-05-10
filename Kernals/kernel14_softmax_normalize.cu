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

// Kernel 14: normalize softmax probabilities.
// Input: logits [batch x classes], row_max [batch], row_sum [batch]
// Output: probs [batch x classes]
__global__ void softmax_normalize_kernel(const float* __restrict__ input,
                                         const float* __restrict__ row_max,
                                         const float* __restrict__ row_sum,
                                         float* __restrict__ output,
                                         int batch,
                                         int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * classes;
    if (idx >= total) {
        return;
    }

    int row = idx / classes;
    float max_val = row_max[row];
    float denom = row_sum[row];
    float v = input[idx];

    output[idx] = expf(v - max_val) / denom;
}

static void cpu_softmax_normalize(const std::vector<float>& input,
                                  const std::vector<float>& row_max,
                                  const std::vector<float>& row_sum,
                                  std::vector<float>& output,
                                  int batch,
                                  int classes) {
    for (int r = 0; r < batch; ++r) {
        float max_val = row_max[r];
        float denom = row_sum[r];
        for (int c = 0; c < classes; ++c) {
            int idx = r * classes + c;
            output[idx] = std::exp(input[idx] - max_val) / denom;
        }
    }
}

int main() {
    // Test config: batch=128, classes=5.
    const int batch = 128;
    const int classes = 5;
    const int total = batch * classes;

    std::vector<float> h_input(total, 0.0f);
    for (int i = 0; i < total; ++i) {
        h_input[i] = 0.01f * static_cast<float>((i % 11) - 5);
    }

    std::vector<float> h_row_max(batch, 0.0f);
    for (int r = 0; r < batch; ++r) {
        float max_val = h_input[r * classes];
        for (int c = 1; c < classes; ++c) {
            float v = h_input[r * classes + c];
            if (v > max_val) {
                max_val = v;
            }
        }
        h_row_max[r] = max_val;
    }

    std::vector<float> h_row_sum(batch, 0.0f);
    for (int r = 0; r < batch; ++r) {
        float sum = 0.0f;
        float max_val = h_row_max[r];
        for (int c = 0; c < classes; ++c) {
            sum += std::exp(h_input[r * classes + c] - max_val);
        }
        h_row_sum[r] = sum;
    }

    std::vector<float> h_output(total, 0.0f);
    std::vector<float> h_expected(total, 0.0f);

    cpu_softmax_normalize(h_input, h_row_max, h_row_sum, h_expected, batch, classes);

    float* d_input = nullptr;
    float* d_row_max = nullptr;
    float* d_row_sum = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_max, h_row_max.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_sum, h_row_sum.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_max, h_row_max.data(), h_row_max.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_sum, h_row_sum.data(), h_row_sum.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    softmax_normalize_kernel<<<blocks, threads>>>(d_input, d_row_max, d_row_sum, d_output, batch, classes);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < total; ++i) {
        float diff = std::fabs(h_output[i] - h_expected[i]);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 14 softmax normalize: PASS\n";
    } else {
        std::cout << "Kernel 14 softmax normalize: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
