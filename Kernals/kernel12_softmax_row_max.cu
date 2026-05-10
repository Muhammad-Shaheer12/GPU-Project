#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>

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

// Warp-level max reduction using shuffle.
__device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

// Kernel 12: compute per-row max for softmax stability.
// Input: [batch x classes], Output: [batch]
__global__ void softmax_row_max_kernel(const float* __restrict__ input,
                                       float* __restrict__ row_max,
                                       int batch,
                                       int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    // Each thread scans a strided subset of the row.
    float local_max = -FLT_MAX;
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        local_max = fmaxf(local_max, v);
    }

    float max_val = warp_reduce_max(local_max);
    if (tid == 0) {
        row_max[row] = max_val;
    }
}

static void cpu_row_max(const std::vector<float>& input,
                        std::vector<float>& row_max,
                        int batch,
                        int classes) {
    for (int r = 0; r < batch; ++r) {
        float max_val = -FLT_MAX;
        for (int c = 0; c < classes; ++c) {
            float v = input[r * classes + c];
            if (v > max_val) {
                max_val = v;
            }
        }
        row_max[r] = max_val;
    }
}

int main() {
    // Test config: batch=128, classes=5.
    const int batch = 128;
    const int classes = 5;

    std::vector<float> h_input(batch * classes, 0.0f);
    for (int i = 0; i < batch * classes; ++i) {
        h_input[i] = 0.01f * static_cast<float>((i % 11) - 5);
    }

    std::vector<float> h_row_max(batch, 0.0f);
    std::vector<float> h_expected(batch, 0.0f);

    cpu_row_max(h_input, h_expected, batch, classes);

    float* d_input = nullptr;
    float* d_row_max = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_max, h_row_max.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 32;
    const int blocks = batch;

    softmax_row_max_kernel<<<blocks, threads>>>(d_input, d_row_max, batch, classes);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_row_max.data(), d_row_max, h_row_max.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < batch; ++i) {
        float diff = std::fabs(h_row_max[i] - h_expected[i]);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at row " << i << ": got " << h_row_max[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 12 softmax row max: PASS\n";
    } else {
        std::cout << "Kernel 12 softmax row max: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_row_max));

    return ok ? 0 : 1;
}
