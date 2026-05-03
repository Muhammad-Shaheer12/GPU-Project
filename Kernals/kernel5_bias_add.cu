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

static void cpu_bias_add(const std::vector<float>& input,
                         const std::vector<float>& bias,
                         std::vector<float>& output,
                         int rows,
                         int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            output[idx] = input[idx] + bias[c];
        }
    }
}

int main() {
    // Test config: rows = batch, cols = hidden dim (64).
    const int rows = 4;
    const int cols = 64;
    const int total = rows * cols;

    std::vector<float> h_input(total, 0.0f);
    std::vector<float> h_bias(cols, 0.0f);

    for (int i = 0; i < total; ++i) {
        h_input[i] = 0.01f * static_cast<float>(i);
    }
    for (int c = 0; c < cols; ++c) {
        h_bias[c] = 1.0f + 0.1f * static_cast<float>(c);
    }

    std::vector<float> h_output(total, 0.0f);
    std::vector<float> h_expected(total, 0.0f);

    cpu_bias_add(h_input, h_bias, h_expected, rows, cols);

    float* d_input = nullptr;
    float* d_bias = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    bias_add_kernel<<<blocks, threads>>>(d_input, d_bias, d_output, rows, cols);

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
        std::cout << "Kernel 5 bias add: PASS\n";
    } else {
        std::cout << "Kernel 5 bias add: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
