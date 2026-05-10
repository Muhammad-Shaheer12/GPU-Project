#include "common.h"

// Fused bias + leaky ReLU to reduce global memory traffic.
// Uses float4 vectorized loads for performance.
__global__ void bias_leaky_relu_kernel(const float* __restrict__ input,
                                       const float* __restrict__ bias,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols,
                                       float alpha) {
    int total = rows * cols;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx4 * 4;

    if (cols % 4 == 0 && base + 3 < total) {
        int col_base = base % cols;
        float4 in = reinterpret_cast<const float4*>(input + base)[0];
        float4 b = reinterpret_cast<const float4*>(bias + col_base)[0];

        float x0 = in.x + b.x;
        float x1 = in.y + b.y;
        float x2 = in.z + b.z;
        float x3 = in.w + b.w;

        float4 out_vec;
        out_vec.x = (x0 >= 0.0f) ? x0 : (alpha * x0);
        out_vec.y = (x1 >= 0.0f) ? x1 : (alpha * x1);
        out_vec.z = (x2 >= 0.0f) ? x2 : (alpha * x2);
        out_vec.w = (x3 >= 0.0f) ? x3 : (alpha * x3);

        reinterpret_cast<float4*>(output + base)[0] = out_vec;
        return;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int col = idx % cols;
    float x = input[idx] + bias[col];
    output[idx] = (x >= 0.0f) ? x : (alpha * x);
}
