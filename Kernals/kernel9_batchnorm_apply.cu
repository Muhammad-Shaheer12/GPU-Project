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

// Kernel 9: apply batch normalization.
// y = (x - mean) / sqrt(var + eps) * gamma + beta
__global__ void batchnorm_apply_kernel(const float* __restrict__ input,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ var,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ output,
                                       int batch,
                                       int features,
                                       float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * features;
    if (idx >= total) {
        return;
    }

    int f = idx % features;
    float x = input[idx];
    float norm = (x - mean[f]) / sqrtf(var[f] + eps);
    output[idx] = norm * gamma[f] + beta[f];
}

static void cpu_batchnorm_apply(const std::vector<float>& input,
                                const std::vector<float>& mean,
                                const std::vector<float>& var,
                                const std::vector<float>& gamma,
                                const std::vector<float>& beta,
                                std::vector<float>& output,
                                int batch,
                                int features,
                                float eps) {
    for (int i = 0; i < batch; ++i) {
        for (int f = 0; f < features; ++f) {
            int idx = i * features + f;
            float norm = (input[idx] - mean[f]) / std::sqrt(var[f] + eps);
            output[idx] = norm * gamma[f] + beta[f];
        }
    }
}

int main() {
    // Test config: batch=256, features=64.
    const int batch = 256;
    const int features = 64;
    const float eps = 1e-5f;

    std::vector<float> h_input(batch * features, 0.0f);
    for (int i = 0; i < batch; ++i) {
        for (int f = 0; f < features; ++f) {
            h_input[i * features + f] = 0.001f * static_cast<float>(i * 17 + f * 3);
        }
    }

    std::vector<float> h_mean(features, 0.0f);
    std::vector<float> h_var(features, 0.0f);
    std::vector<float> h_gamma(features, 1.0f);
    std::vector<float> h_beta(features, 0.0f);

    // Simple mean/var for test.
    for (int f = 0; f < features; ++f) {
        float sum = 0.0f;
        for (int i = 0; i < batch; ++i) {
            sum += h_input[i * features + f];
        }
        h_mean[f] = sum / static_cast<float>(batch);
    }
    for (int f = 0; f < features; ++f) {
        float sum = 0.0f;
        for (int i = 0; i < batch; ++i) {
            float diff = h_input[i * features + f] - h_mean[f];
            sum += diff * diff;
        }
        h_var[f] = sum / static_cast<float>(batch);
    }

    std::vector<float> h_output(batch * features, 0.0f);
    std::vector<float> h_expected(batch * features, 0.0f);

    cpu_batchnorm_apply(h_input, h_mean, h_var, h_gamma, h_beta, h_expected, batch, features, eps);

    float* d_input = nullptr;
    float* d_mean = nullptr;
    float* d_var = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, h_mean.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, h_var.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, h_gamma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, h_beta.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean.data(), h_mean.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_var, h_var.data(), h_var.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int total = batch * features;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    batchnorm_apply_kernel<<<blocks, threads>>>(
        d_input,
        d_mean,
        d_var,
        d_gamma,
        d_beta,
        d_output,
        batch,
        features,
        eps);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < total; ++i) {
        float diff = std::fabs(h_output[i] - h_expected[i]);
        if (diff > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 9 batchnorm apply: PASS\n";
    } else {
        std::cout << "Kernel 9 batchnorm apply: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_var));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
