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

// Warp-level argmax using shuffle.
__device__ void warp_reduce_argmax(float& v, int& idx) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_v = __shfl_down_sync(0xFFFFFFFF, v, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_v > v) {
            v = other_v;
            idx = other_idx;
        }
    }
}

// Kernel 15: argmax per row.
// Input: probs [batch x classes], Output: argmax indices [batch]
__global__ void argmax_kernel(const float* __restrict__ input,
                              int* __restrict__ output,
                              int batch,
                              int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        if (v > local_max) {
            local_max = v;
            local_idx = c;
        }
    }

    warp_reduce_argmax(local_max, local_idx);
    if (tid == 0) {
        output[row] = local_idx;
    }
}

static void cpu_argmax(const std::vector<float>& input,
                       std::vector<int>& output,
                       int batch,
                       int classes) {
    for (int r = 0; r < batch; ++r) {
        float max_val = -FLT_MAX;
        int max_idx = 0;
        for (int c = 0; c < classes; ++c) {
            float v = input[r * classes + c];
            if (v > max_val) {
                max_val = v;
                max_idx = c;
            }
        }
        output[r] = max_idx;
    }
}

int main() {
    // Test config: batch=128, classes=5.
    const int batch = 128;
    const int classes = 5;

    std::vector<float> h_input(batch * classes, 0.0f);
    for (int r = 0; r < batch; ++r) {
        for (int c = 0; c < classes; ++c) {
            h_input[r * classes + c] = 0.01f * static_cast<float>((r + 1) * (c + 1));
        }
        h_input[r * classes + (r % classes)] += 1.0f; // force a clear max
    }

    std::vector<int> h_output(batch, 0);
    std::vector<int> h_expected(batch, 0);

    cpu_argmax(h_input, h_expected, batch, classes);

    float* d_input = nullptr;
    int* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 32;
    const int blocks = batch;

    argmax_kernel<<<blocks, threads>>>(d_input, d_output, batch, classes);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < batch; ++i) {
        if (h_output[i] != h_expected[i]) {
            std::cerr << "Mismatch at row " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 15 argmax: PASS\n";
    } else {
        std::cout << "Kernel 15 argmax: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
