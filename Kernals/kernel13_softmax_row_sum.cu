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

// Kernel 13: compute per-row sum of exp(x - row_max).
// Input: logits [batch x classes], row_max [batch], Output: row_sum [batch]
__global__ void softmax_row_sum_kernel(const float* __restrict__ input,
                                       const float* __restrict__ row_max,
                                       float* __restrict__ row_sum,
                                       int batch,
                                       int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    float max_val = row_max[row];
    float local_sum = 0.0f;
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        local_sum += expf(v - max_val);
    }

    extern __shared__ float sdata[];
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sum[row] = sdata[0];
    }
}

static void cpu_row_sum(const std::vector<float>& input,
                        const std::vector<float>& row_max,
                        std::vector<float>& row_sum,
                        int batch,
                        int classes) {
    for (int r = 0; r < batch; ++r) {
        float sum = 0.0f;
        float max_val = row_max[r];
        for (int c = 0; c < classes; ++c) {
            sum += std::exp(input[r * classes + c] - max_val);
        }
        row_sum[r] = sum;
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
    std::vector<float> h_expected(batch, 0.0f);

    cpu_row_sum(h_input, h_row_max, h_expected, batch, classes);

    float* d_input = nullptr;
    float* d_row_max = nullptr;
    float* d_row_sum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_max, h_row_max.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_sum, h_row_sum.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_max, h_row_max.data(), h_row_max.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = batch;
    const size_t shared_bytes = threads * sizeof(float);

    softmax_row_sum_kernel<<<blocks, threads, shared_bytes>>>(d_input, d_row_max, d_row_sum, batch, classes);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_row_sum.data(), d_row_sum, h_row_sum.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < batch; ++i) {
        float diff = std::fabs(h_row_sum[i] - h_expected[i]);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at row " << i << ": got " << h_row_sum[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 13 softmax row sum: PASS\n";
    } else {
        std::cout << "Kernel 13 softmax row sum: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));

    return ok ? 0 : 1;
}
