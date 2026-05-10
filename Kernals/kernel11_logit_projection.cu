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

// Kernel 11: final logit projection (matrix-vector per batch row).
// Input: [batch x hidden], Weights: [hidden x classes], Output: [batch x classes]
__global__ void logit_projection_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weights,
                                        float* __restrict__ output,
                                        int batch,
                                        int hidden,
                                        int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * classes;
    if (idx >= total) {
        return;
    }

    int row = idx / classes;
    int cls = idx - row * classes;

    float sum = 0.0f;
    const float* in_row = input + row * hidden;
    const float* w_col = weights + cls; // column-major access via stride

    for (int k = 0; k < hidden; ++k) {
        sum += in_row[k] * w_col[k * classes];
    }

    output[idx] = sum;
}

static void cpu_logit_projection(const std::vector<float>& input,
                                 const std::vector<float>& weights,
                                 std::vector<float>& output,
                                 int batch,
                                 int hidden,
                                 int classes) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < classes; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                sum += input[b * hidden + k] * weights[k * classes + c];
            }
            output[b * classes + c] = sum;
        }
    }
}

int main() {
    // Test config: batch=128, hidden=128, classes=5.
    const int batch = 128;
    const int hidden = 128;
    const int classes = 5;
    const int total = batch * classes;

    std::vector<float> h_input(batch * hidden, 0.0f);
    std::vector<float> h_weights(hidden * classes, 0.0f);

    for (int i = 0; i < batch * hidden; ++i) {
        h_input[i] = 0.001f * static_cast<float>(i % 101);
    }
    for (int i = 0; i < hidden * classes; ++i) {
        h_weights[i] = 0.002f * static_cast<float>(i % 97);
    }

    std::vector<float> h_output(total, 0.0f);
    std::vector<float> h_expected(total, 0.0f);

    cpu_logit_projection(h_input, h_weights, h_expected, batch, hidden, classes);

    float* d_input = nullptr;
    float* d_weights = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    logit_projection_kernel<<<blocks, threads>>>(
        d_input,
        d_weights,
        d_output,
        batch,
        hidden,
        classes);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < total; ++i) {
        float diff = std::fabs(h_output[i] - h_expected[i]);
        if (diff > 1e-3f) {
            std::cerr << "Mismatch at " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 11 logit projection: PASS\n";
    } else {
        std::cout << "Kernel 11 logit projection: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
