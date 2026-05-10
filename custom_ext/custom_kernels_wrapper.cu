#include <cuda_runtime.h>

// ============================================================
// Forward declarations of all __global__ kernel functions
// ============================================================

// K1: pad_truncate
__global__ void pad_truncate_kernel(const int* input_tokens,
                                    const int* input_lengths,
                                    int* output_tokens,
                                    int batch,
                                    int input_stride,
                                    int fixed_len,
                                    int pad_token);

// K2: embedding_lookup
__global__ void embedding_lookup_kernel(const int* __restrict__ tokens,
                                        const float* __restrict__ embedding,
                                        float* __restrict__ output,
                                        int total_tokens,
                                        int dim,
                                        int vocab,
                                        int unk_id);

// K3: positional_encoding
__global__ void positional_encoding_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int total_tokens,
                                           int dim);

// K4: weighted_mean_pooling
__global__ void weighted_mean_pooling_kernel(const float* __restrict__ input,
                                             const float* __restrict__ weights,
                                             float* __restrict__ output,
                                             int batch,
                                             int seq_len,
                                             int dim);

// K5: bias_add
__global__ void bias_add_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int rows,
                                int cols);

// K6: leaky_relu
__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int total,
                                  float alpha);

// K7: batchnorm_mean
__global__ void batchnorm_mean_kernel(const float* __restrict__ input,
                                      float* __restrict__ mean,
                                      int batch,
                                      int features);

// K8: batchnorm_var
__global__ void batchnorm_var_kernel(const float* __restrict__ input,
                                     const float* __restrict__ mean,
                                     float* __restrict__ var,
                                     int batch,
                                     int features);

// K9: batchnorm_apply
__global__ void batchnorm_apply_kernel(const float* __restrict__ input,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ var,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ output,
                                       int batch,
                                       int features,
                                       float eps);

// K10: gemm_tiled
constexpr int TILE = 16;
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int K,
                                  int N);

// K11: logit_projection
__global__ void logit_projection_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weights,
                                        float* __restrict__ output,
                                        int batch,
                                        int hidden,
                                        int classes);

// K12: softmax_row_max
__global__ void softmax_row_max_kernel(const float* __restrict__ input,
                                       float* __restrict__ row_max,
                                       int batch,
                                       int classes);

// K13: softmax_row_sum
__global__ void softmax_row_sum_kernel(const float* __restrict__ input,
                                       const float* __restrict__ row_max,
                                       float* __restrict__ row_sum,
                                       int batch,
                                       int classes);

// K14: softmax_normalize
__global__ void softmax_normalize_kernel(const float* __restrict__ input,
                                         const float* __restrict__ row_max,
                                         const float* __restrict__ row_sum,
                                         float* __restrict__ output,
                                         int batch,
                                         int classes);

// K15: argmax
__global__ void argmax_kernel(const float* __restrict__ input,
                              int* __restrict__ output,
                              int batch,
                              int classes);


// ============================================================
// Launch wrapper functions (called from custom_ops.cpp)
// ============================================================

void launch_pad_truncate(const int* input_tokens,
                         const int* input_lengths,
                         int* output_tokens,
                         int batch,
                         int input_stride,
                         int fixed_len,
                         int pad_token) {
    int total = batch * fixed_len;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    pad_truncate_kernel<<<blocks, threads>>>(
        input_tokens, input_lengths, output_tokens,
        batch, input_stride, fixed_len, pad_token);
}

void launch_embedding_lookup(const int* tokens,
                              const float* embedding,
                              float* output,
                              int total_tokens,
                              int dim,
                              int vocab,
                              int unk_id) {
    dim3 block(16, 8, 1); // 16 lanes * 4 dims = 64 dims; 8 tokens per block
    dim3 grid(1, (total_tokens + block.y - 1) / block.y, 1);
    embedding_lookup_kernel<<<grid, block>>>(
        tokens, embedding, output, total_tokens, dim, vocab, unk_id);
}

void launch_positional_encoding(const float* input,
                                 float* output,
                                 int total_tokens,
                                 int dim) {
    const int threads = 256;
    int total = total_tokens * dim;
    const int blocks = (total + threads - 1) / threads;
    positional_encoding_kernel<<<blocks, threads>>>(input, output, total_tokens, dim);
}

void launch_weighted_mean_pooling(const float* input,
                                   const float* weights,
                                   float* output,
                                   int batch,
                                   int seq_len,
                                   int dim) {
    dim3 block(seq_len, 1, 1);
    dim3 grid(batch, dim, 1);
    size_t shared_bytes = static_cast<size_t>(seq_len) * sizeof(float) * 2;
    weighted_mean_pooling_kernel<<<grid, block, shared_bytes>>>(
        input, weights, output, batch, seq_len, dim);
}

void launch_bias_add(const float* input,
                      const float* bias,
                      float* output,
                      int rows,
                      int cols) {
    int total = rows * cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads>>>(input, bias, output, rows, cols);
}

void launch_leaky_relu(const float* input,
                        float* output,
                        int total,
                        float alpha) {
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, total, alpha);
}

void launch_batchnorm_mean(const float* input,
                            float* mean,
                            int batch,
                            int features) {
    const int threads = 256;
    const int blocks = features;
    batchnorm_mean_kernel<<<blocks, threads>>>(input, mean, batch, features);
}

void launch_batchnorm_var(const float* input,
                           const float* mean,
                           float* var,
                           int batch,
                           int features) {
    const int threads = 256;
    const int blocks = features;
    batchnorm_var_kernel<<<blocks, threads>>>(input, mean, var, batch, features);
}

void launch_batchnorm_apply(const float* input,
                              const float* mean,
                              const float* var,
                              const float* gamma,
                              const float* beta,
                              float* output,
                              int batch,
                              int features,
                              float eps) {
    int total = batch * features;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    batchnorm_apply_kernel<<<blocks, threads>>>(
        input, mean, var, gamma, beta, output, batch, features, eps);
}

void launch_gemm_tiled(const float* A,
                        const float* B,
                        float* C,
                        int M,
                        int K,
                        int N) {
    dim3 block(TILE, TILE, 1);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
    gemm_tiled_kernel<<<grid, block>>>(A, B, C, M, K, N);
}

void launch_logit_projection(const float* input,
                               const float* weights,
                               float* output,
                               int batch,
                               int hidden,
                               int classes) {
    int total = batch * classes;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    logit_projection_kernel<<<blocks, threads>>>(
        input, weights, output, batch, hidden, classes);
}

void launch_softmax_row_max(const float* input,
                              float* row_max,
                              int batch,
                              int classes) {
    const int threads = 32;
    const int blocks = batch;
    softmax_row_max_kernel<<<blocks, threads>>>(
        input, row_max, batch, classes);
}

void launch_softmax_row_sum(const float* input,
                              const float* row_max,
                              float* row_sum,
                              int batch,
                              int classes) {
    const int threads = 32;
    const int blocks = batch;
    softmax_row_sum_kernel<<<blocks, threads>>>(
        input, row_max, row_sum, batch, classes);
}

void launch_softmax_normalize(const float* input,
                                const float* row_max,
                                const float* row_sum,
                                float* output,
                                int batch,
                                int classes) {
    int total = batch * classes;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    softmax_normalize_kernel<<<blocks, threads>>>(
        input, row_max, row_sum, output, batch, classes);
}

void launch_argmax(const float* input,
                    int* output,
                    int batch,
                    int classes) {
    const int threads = 32;
    const int blocks = batch;
    argmax_kernel<<<blocks, threads>>>(
        input, output, batch, classes);
}
