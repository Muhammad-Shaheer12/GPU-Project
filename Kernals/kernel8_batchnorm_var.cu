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

// Warp-level reduction for sum using shuffle.
__device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    }
    return v;
}

// Kernel 8: compute variance per feature across the batch.
// Input shape: [batch, features], row-major.
// Mean is precomputed by Kernel 7.
__global__ void batchnorm_var_kernel(const float* __restrict__ input,
                                     const float* __restrict__ mean,
                                     float* __restrict__ var,
                                     int batch,
                                     int features) {
    int feature = blockIdx.x;
    int tid = threadIdx.x;

    if (feature >= features) {
        return;
    }

    float sum = 0.0f;
    float m = mean[feature];
    for (int i = tid; i < batch; i += blockDim.x) {
        float diff = input[i * features + feature] - m;
        sum += diff * diff;
    }

    float warp_sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            var[feature] = block_sum / static_cast<float>(batch);
        }
    }
}

static void cpu_batchnorm_var(const std::vector<float>& input,
                              const std::vector<float>& mean,
                              std::vector<float>& var,
                              int batch,
                              int features) {
    for (int f = 0; f < features; ++f) {
        float sum = 0.0f;
        for (int i = 0; i < batch; ++i) {
            float diff = input[i * features + f] - mean[f];
            sum += diff * diff;
        }
        var[f] = sum / static_cast<float>(batch);
    }
}

int main() {
    // Test config: batch=256, features=64.
    const int batch = 256;
    const int features = 64;

    std::vector<float> h_input(batch * features, 0.0f);
    for (int i = 0; i < batch; ++i) {
        for (int f = 0; f < features; ++f) {
            h_input[i * features + f] = 0.001f * static_cast<float>(i * 17 + f * 3);
        }
    }

    std::vector<float> h_mean(features, 0.0f);
    std::vector<float> h_var(features, 0.0f);
    std::vector<float> h_expected(features, 0.0f);

    // CPU mean first, then variance.
    for (int f = 0; f < features; ++f) {
        float sum = 0.0f;
        for (int i = 0; i < batch; ++i) {
            sum += h_input[i * features + f];
        }
        h_mean[f] = sum / static_cast<float>(batch);
    }

    cpu_batchnorm_var(h_input, h_mean, h_expected, batch, features);

    float* d_input = nullptr;
    float* d_mean = nullptr;
    float* d_var = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, h_mean.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, h_var.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean.data(), h_mean.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = features;

    batchnorm_var_kernel<<<blocks, threads>>>(d_input, d_mean, d_var, batch, features);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_var.data(), d_var, h_var.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int f = 0; f < features; ++f) {
        float diff = std::fabs(h_var[f] - h_expected[f]);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at feature " << f << ": got " << h_var[f]
                      << ", expected " << h_expected[f] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 8 batchnorm var: PASS\n";
    } else {
        std::cout << "Kernel 8 batchnorm var: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_var));

    return ok ? 0 : 1;
}
