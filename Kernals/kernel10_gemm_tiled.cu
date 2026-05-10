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

// Kernel 10: tiled GEMM C = A * B
// A: [M x K], B: [K x N], C: [M x N]
constexpr int TILE = 16;

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int K,
                                  int N) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    // Loop over tiles of K.
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles.
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

static void cpu_gemm(const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C,
                     int M,
                     int K,
                     int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // Test config aligned with your pipeline: M=batch, K=64, N=128.
    const int M = 128;
    const int K = 64;
    const int N = 128;

    std::vector<float> h_A(M * K, 0.0f);
    std::vector<float> h_B(K * N, 0.0f);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 0.001f * static_cast<float>(i % 97);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 0.002f * static_cast<float>(i % 89);
    }

    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_expected(M * N, 0.0f);

    cpu_gemm(h_A, h_B, h_expected, M, K, N);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE, 1);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);

    gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float diff = std::fabs(h_C[i] - h_expected[i]);
        if (diff > 1e-3f) {
            std::cerr << "Mismatch at " << i << ": got " << h_C[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 10 GEMM tiled: PASS\n";
    } else {
        std::cout << "Kernel 10 GEMM tiled: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return ok ? 0 : 1;
}
