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

// Kernel 7: compute mean per feature across the batch.
// Input shape: [batch, features], row-major.
__global__ void batchnorm_mean_kernel(const float* __restrict__ input,
                                      float* __restrict__ mean,
                                      int batch,
                                      int features) {
    int feature = blockIdx.x;
    int tid = threadIdx.x;

    if (feature >= features) {
        return;
    }

    float sum = 0.0f;
    for (int i = tid; i < batch; i += blockDim.x) {
        sum += input[i * features + feature];
    }

    // Reduce within each warp.
    float warp_sum = warp_reduce_sum(sum);

    // One value per warp goes to shared memory.
    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction by the first warp.
    float block_sum = 0.0f;
    if (warp_id == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            mean[feature] = block_sum / static_cast<float>(batch);
        }
    }
}

static void cpu_batchnorm_mean(const std::vector<float>& input,
                               std::vector<float>& mean,
                               int batch,
                               int features) {
    for (int f = 0; f < features; ++f) {
        float sum = 0.0f;
        for (int i = 0; i < batch; ++i) {
            sum += input[i * features + f];
        }
        mean[f] = sum / static_cast<float>(batch);
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
    std::vector<float> h_expected(features, 0.0f);

    cpu_batchnorm_mean(h_input, h_expected, batch, features);

    float* d_input = nullptr;
    float* d_mean = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, h_mean.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = features;

    batchnorm_mean_kernel<<<blocks, threads>>>(d_input, d_mean, batch, features);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, h_mean.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int f = 0; f < features; ++f) {
        float diff = std::fabs(h_mean[f] - h_expected[f]);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at feature " << f << ": got " << h_mean[f]
                      << ", expected " << h_expected[f] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 7 batchnorm mean: PASS\n";
    } else {
        std::cout << "Kernel 7 batchnorm mean: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_mean));

    return ok ? 0 : 1;
}
