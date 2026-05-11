#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include "bridge.hpp"

// ============================================================
// Unity Build: Include all kernels directly
// This simplifies the build system and allows for better optimization.
// ============================================================
#include "kernel1_pad_truncate.cu"
#include "kernel2_embedding_lookup.cu"
#include "kernel3_positional_encoding.cu"
#include "kernel4_weighted_mean_pooling.cu"
#include "kernel5_bias_add.cu"
#include "kernel6_leaky_relu.cu"
#include "kernel7_batchnorm_mean.cu"
#include "kernel8_batchnorm_var.cu"
#include "kernel9_batchnorm_apply.cu"
#include "kernel10_gemm_tiled.cu"
#include "kernel11_logit_projection.cu"
#include "kernel12_softmax_row_max.cu"
#include "kernel13_softmax_row_sum.cu"
#include "kernel14_softmax_normalize.cu"
#include "kernel15_argmax.cu"
#include "kernel16_fused_bias_relu.cu"
#include "kernel17_fused_softmax.cu"

// cuBLAS handle management
static cublasHandle_t global_handle = nullptr;

static void ensure_cublas_handle() {
    if (global_handle == nullptr) {
        if (cublasCreate(&global_handle) != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Failed to create cuBLAS handle\n";
            std::exit(1);
        }
        cublasSetMathMode(global_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
}

// ============================================================
// Launch Wrappers
// ============================================================

void launch_pad_truncate(const int* input, const int* lengths, int* output, int batch, int stride, int fixed_len, int pad_token) {
    const int threads = 256;
    int blocks = (batch * fixed_len + threads - 1) / threads;
    pad_truncate_kernel<<<blocks, threads>>>(input, lengths, output, batch, stride, fixed_len, pad_token);
}

void launch_embedding_lookup(const int* tokens, const float* embedding, float* output, int total_tokens, int dim, int vocab, int unk_id) {
    const int threads = 256;
    int blocks = (total_tokens * dim + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads>>>(tokens, embedding, output, total_tokens, dim, vocab, unk_id);
}

void launch_positional_encoding(const float* input, float* output, int total_tokens, int dim) {
    const int threads = 256;
    int blocks = (total_tokens * dim + threads - 1) / threads;
    positional_encoding_kernel<<<blocks, threads>>>(input, output, total_tokens, dim);
}

void launch_weighted_mean_pooling(const float* input, const float* weights, float* output, int batch, int seq_len, int dim) {
    const int threads = 256;
    int blocks = (batch * dim + threads - 1) / threads;
    weighted_mean_pooling_kernel<<<blocks, threads>>>(input, weights, output, batch, seq_len, dim);
}

void launch_bias_add(const float* input, const float* bias, float* output, int rows, int cols) {
    const int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads>>>(input, bias, output, rows, cols);
}

void launch_leaky_relu(const float* input, float* output, int total, float alpha) {
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, total, alpha);
}

void launch_batchnorm_mean(const float* input, float* mean, int batch, int features) {
    batchnorm_mean_kernel<<<features, 256>>>(input, mean, batch, features);
}

void launch_batchnorm_var(const float* input, const float* mean, float* var, int batch, int features) {
    batchnorm_var_kernel<<<features, 256>>>(input, mean, var, batch, features);
}

void launch_batchnorm_apply(const float* input, const float* mean, const float* var, const float* gamma, const float* beta, float* output, int batch, int features, float eps) {
    const int threads = 256;
    int blocks = (batch * features + threads - 1) / threads;
    batchnorm_apply_kernel<<<blocks, threads>>>(input, mean, var, gamma, beta, output, batch, features, eps);
}

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    ensure_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(global_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
}

void launch_logit_projection(const float* input, const float* weight, float* output, int batch, int hidden, int classes) {
    const int threads = 32;
    logit_projection_kernel<<<batch, threads>>>(input, weight, output, batch, hidden, classes);
}

void launch_softmax_row_max(const float* input, float* output, int batch, int classes) {
    softmax_row_max_kernel<<<batch, 32>>>(input, output, batch, classes);
}

void launch_softmax_row_sum(const float* input, const float* row_max, float* output, int batch, int classes) {
    softmax_row_sum_kernel<<<batch, 32>>>(input, row_max, output, batch, classes);
}

void launch_softmax_normalize(const float* input, const float* row_max, const float* row_sum, float* output, int batch, int classes) {
    const int threads = 256;
    int blocks = (batch * classes + threads - 1) / threads;
    softmax_normalize_kernel<<<blocks, threads>>>(input, row_max, row_sum, output, batch, classes);
}

void launch_argmax(const float* input, int* output, int batch, int classes) {
    argmax_kernel<<<batch, 32>>>(input, output, batch, classes);
}

void launch_fused_bias_leaky_relu(const float* input, const float* bias, float* output, int rows, int cols, float alpha) {
    const int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    bias_leaky_relu_kernel<<<blocks, threads>>>(input, bias, output, rows, cols, alpha);
}

void launch_fused_softmax(const float* input, float* output, int batch, int classes) {
    softmax_fused_kernel<<<batch, 32, 2 * sizeof(float)>>>(input, output, batch, classes);
}
