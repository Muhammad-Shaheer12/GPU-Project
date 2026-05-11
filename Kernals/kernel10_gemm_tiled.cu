#include "common.h"

/**
 * Kernel 10: Advanced Register-Tiled GEMM (C = A * B)
 * --------------------------------------------------
 * This implementation uses thread-level register tiling (4x4) and shared 
 * memory tiling (64x64) to achieve high occupancy and arithmetic intensity.
 * 
 * Performance features:
 * 1. Register Tiling: Each thread calculates 16 output elements in registers.
 * 2. Shared Memory: Cooperative loading to reduce global memory traffic.
 * 3. Loop Unrolling: Pragma unroll for reduced branch overhead.
 */

// Tiling Configuration
const int BM = 64; // Block height in A and C
const int BN = 64; // Block width in B and C
const int BK = 8;  // K-dimension tile size
const int TM = 4;  // Elements per thread (row)
const int TN = 4;  // Elements per thread (col)

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int K,
                                  int N) {
    // 1. Static Allocation
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float accum[TM][TN] = {0.0f};

    // Thread/Block indices
    int tx = threadIdx.x; // range [0, 15]
    int ty = threadIdx.y; // range [0, 15]
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 2. Iterate over the K dimension in blocks of BK
    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        
        // Cooperative Load from A -> As
        // We have 256 threads loading 64x8 = 512 elements (2 elements per thread)
        int tid = ty * 16 + tx;
        int a_inner_row = tid / BK;
        int a_inner_col = tid % BK;
        int a_global_row = by * BM + a_inner_row;
        int a_global_col = k_offset + a_inner_col;
        
        if (a_global_row < M && a_global_col < K)
            As[a_inner_row][a_inner_col] = A[a_global_row * K + a_global_col];
        else
            As[a_inner_row][a_inner_col] = 0.0f;

        // Cooperative Load from B -> Bs
        // We have 256 threads loading 8x64 = 512 elements (2 elements per thread)
        int b_inner_row = tid / BN;
        int b_inner_col = tid % BN;
        int b_global_row = k_offset + b_inner_row;
        int b_global_col = bx * BN + b_inner_col;

        if (b_global_row < K && b_global_col < N)
            Bs[b_inner_row][b_inner_col] = B[b_global_row * N + b_global_col];
        else
            Bs[b_inner_row][b_inner_col] = 0.0f;

        __syncthreads();

        // 3. Compute 4x4 tile per thread using registers
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) a_frag[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // 4. Store results to Global Memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int out_row = by * BM + ty * TM + i;
            int out_col = bx * BN + tx * TN + j;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = accum[i][j];
            }
        }
    }
}
