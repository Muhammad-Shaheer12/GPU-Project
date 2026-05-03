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

static void cpu_leaky_relu(const std::vector<float>& input,
                           std::vector<float>& output,
                           int total,
                           float alpha) {
    for (int i = 0; i < total; ++i) {
        float x = input[i];
        output[i] = (x >= 0.0f) ? x : (alpha * x);
    }
}

int main() {
    // Test config: total = rows * cols from previous layer.
    const int rows = 4;
    const int cols = 64;
    const int total = rows * cols;
    const float alpha = 0.01f;

    std::vector<float> h_input(total, 0.0f);
    for (int i = 0; i < total; ++i) {
        float base = 0.02f * static_cast<float>(i);
        h_input[i] = (i % 5 == 0) ? -base : base; // mix positive and negative
    }

    std::vector<float> h_output(total, 0.0f);
    std::vector<float> h_expected(total, 0.0f);

    cpu_leaky_relu(h_input, h_expected, total, alpha);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(d_input, d_output, total, alpha);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < total; ++i) {
        float diff = std::fabs(h_output[i] - h_expected[i]);
        if (diff > 1e-6f) {
            std::cerr << "Mismatch at " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 6 leaky ReLU: PASS\n";
    } else {
        std::cout << "Kernel 6 leaky ReLU: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
