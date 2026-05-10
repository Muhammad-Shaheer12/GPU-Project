#include <cuda_runtime.h>
#include <cfloat>

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

    extern __shared__ float s_vals[];
    int* s_idx = reinterpret_cast<int*>(s_vals + blockDim.x);

    s_vals[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = s_vals[tid + stride];
            int other_idx = s_idx[tid + stride];
            if (other_val > s_vals[tid]) {
                s_vals[tid] = other_val;
                s_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = s_idx[0];
    }
}
