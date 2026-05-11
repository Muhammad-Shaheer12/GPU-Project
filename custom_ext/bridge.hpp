#pragma once

// ============================================================
// Forward declarations of launch wrappers
// These bridge the PyBind11 C++ code to the CUDA kernels.
// ============================================================

void launch_pad_truncate(const int* input_tokens, const int* input_lengths,
                         int* output_tokens, int batch, int input_stride,
                         int fixed_len, int pad_token);

void launch_embedding_lookup(const int* tokens, const float* embedding,
                              float* output, int total_tokens, int dim,
                              int vocab, int unk_id);

void launch_positional_encoding(const float* input, float* output,
                                 int total_tokens, int dim);

void launch_weighted_mean_pooling(const float* input, const float* weights,
                                   float* output, int batch, int seq_len, int dim);

void launch_bias_add(const float* input, const float* bias,
                      float* output, int rows, int cols);

void launch_leaky_relu(const float* input, float* output, int total, float alpha);

void launch_batchnorm_mean(const float* input, float* mean, int batch, int features);

void launch_batchnorm_var(const float* input, const float* mean,
                           float* var, int batch, int features);

void launch_batchnorm_apply(const float* input, const float* mean,
                              const float* var, const float* gamma,
                              const float* beta, float* output,
                              int batch, int features, float eps);

void launch_gemm_tiled(const float* A, const float* B, float* C, int M, int K, int N);
void launch_gemm_cublas(const float* A, const float* B, float* C, int M, int K, int N);
void launch_gemm_custom(const float* A, const float* B, float* C, int M, int K, int N);

void launch_logit_projection(const float* input, const float* weights,
                               float* output, int batch, int hidden, int classes);

void launch_softmax_row_max(const float* input, float* row_max,
                              int batch, int classes);

void launch_softmax_row_sum(const float* input, const float* row_max,
                              float* row_sum, int batch, int classes);

void launch_softmax_normalize(const float* input, const float* row_max,
                                const float* row_sum, float* output,
                                int batch, int classes);

void launch_argmax(const float* input, int* output, int batch, int classes);

void launch_fused_bias_leaky_relu(const float* input, const float* bias, 
                                  float* output, int rows, int cols, float alpha);

void launch_fused_softmax(const float* input, float* output, int batch, int classes);
