#include "common.h"
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <cfloat>
#include <chrono>

#define PIPELINE_BUILD

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

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "CUBLAS error: " << status                       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#ifndef USE_CUBLAS_TF32
#define USE_CUBLAS_TF32 1
#endif

#ifndef USE_FUSED_SOFTMAX
#define USE_FUSED_SOFTMAX 1
#endif

static void cpu_reference_pipeline(const std::vector<int>& input_tokens,
                                   const std::vector<int>& input_lengths,
                                   const std::vector<float>& embedding,
                                   const std::vector<float>& bias,
                                   const std::vector<float>& gamma,
                                   const std::vector<float>& beta,
                                   const std::vector<float>& w_hidden,
                                   const std::vector<float>& w_logits,
                                   std::vector<int>& out_argmax,
                                   int batch,
                                   int input_stride,
                                   int seq_len,
                                   int dim,
                                   int hidden,
                                   int classes,
                                   int vocab,
                                   int pad_token,
                                   int unk_id,
                                   float alpha,
                                   float eps) {
    std::vector<int> tokens(batch * seq_len, pad_token);
    for (int s = 0; s < batch; ++s) {
        int len = input_lengths[s];
        int capped = len < seq_len ? len : seq_len;
        for (int p = 0; p < seq_len; ++p) {
            if (p < capped) {
                tokens[s * seq_len + p] = input_tokens[s * input_stride + p];
            } else {
                tokens[s * seq_len + p] = pad_token;
            }
        }
    }

    std::vector<float> embed(batch * seq_len * dim, 0.0f);
    for (int t = 0; t < batch * seq_len; ++t) {
        int token = tokens[t];
        if (token < 0 || token >= vocab) {
            token = unk_id;
        }
        for (int d = 0; d < dim; ++d) {
            embed[t * dim + d] = embedding[token * dim + d];
        }
    }

    std::vector<float> pos(batch * seq_len * dim, 0.0f);
    for (int t = 0; t < batch * seq_len; ++t) {
        for (int d = 0; d < dim; ++d) {
            int pair = d / 2;
            float exponent = (2.0f * pair) / static_cast<float>(dim);
            float denom = std::exp(std::log(10000.0f) * exponent);
            float angle = static_cast<float>(t) / denom;
            float pe = (d % 2 == 0) ? std::sin(angle) : std::cos(angle);
            pos[t * dim + d] = embed[t * dim + d] + pe;
        }
    }

    std::vector<float> weights(batch * seq_len, 0.0f);
    for (int s = 0; s < batch; ++s) {
        for (int t = 0; t < seq_len; ++t) {
            weights[s * seq_len + t] = (t < input_lengths[s]) ? 1.0f : 0.0f;
        }
    }

    std::vector<float> pooled(batch * dim, 0.0f);
    for (int s = 0; s < batch; ++s) {
        for (int d = 0; d < dim; ++d) {
            float sum = 0.0f;
            float wsum = 0.0f;
            for (int t = 0; t < seq_len; ++t) {
                float w = weights[s * seq_len + t];
                sum += pos[(s * seq_len + t) * dim + d] * w;
                wsum += w;
            }
            pooled[s * dim + d] = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
        }
    }

    std::vector<float> biased(batch * dim, 0.0f);
    for (int i = 0; i < batch * dim; ++i) {
        biased[i] = pooled[i] + bias[i % dim];
    }

    std::vector<float> activated(batch * dim, 0.0f);
    for (int i = 0; i < batch * dim; ++i) {
        float x = biased[i];
        activated[i] = (x >= 0.0f) ? x : (alpha * x);
    }

    std::vector<float> mean(dim, 0.0f);
    for (int d = 0; d < dim; ++d) {
        float sum = 0.0f;
        for (int s = 0; s < batch; ++s) {
            sum += activated[s * dim + d];
        }
        mean[d] = sum / static_cast<float>(batch);
    }

    std::vector<float> var(dim, 0.0f);
    for (int d = 0; d < dim; ++d) {
        float sum = 0.0f;
        for (int s = 0; s < batch; ++s) {
            float diff = activated[s * dim + d] - mean[d];
            sum += diff * diff;
        }
        var[d] = sum / static_cast<float>(batch);
    }

    std::vector<float> bn_out(batch * dim, 0.0f);
    for (int i = 0; i < batch * dim; ++i) {
        int d = i % dim;
        float norm = (activated[i] - mean[d]) / std::sqrt(var[d] + eps);
        bn_out[i] = norm * gamma[d] + beta[d];
    }

    std::vector<float> hidden_out(batch * hidden, 0.0f);
    for (int r = 0; r < batch; ++r) {
        for (int c = 0; c < hidden; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < dim; ++k) {
                sum += bn_out[r * dim + k] * w_hidden[k * hidden + c];
            }
            hidden_out[r * hidden + c] = sum;
        }
    }

    std::vector<float> logits(batch * classes, 0.0f);
    for (int r = 0; r < batch; ++r) {
        for (int c = 0; c < classes; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                sum += hidden_out[r * hidden + k] * w_logits[k * classes + c];
            }
            logits[r * classes + c] = sum;
        }
    }

    std::vector<float> row_max(batch, -FLT_MAX);
    for (int r = 0; r < batch; ++r) {
        float max_val = logits[r * classes];
        for (int c = 1; c < classes; ++c) {
            float v = logits[r * classes + c];
            if (v > max_val) {
                max_val = v;
            }
        }
        row_max[r] = max_val;
    }

    std::vector<float> row_sum(batch, 0.0f);
    for (int r = 0; r < batch; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < classes; ++c) {
            sum += std::exp(logits[r * classes + c] - row_max[r]);
        }
        row_sum[r] = sum;
    }

    std::vector<float> probs(batch * classes, 0.0f);
    for (int r = 0; r < batch; ++r) {
        for (int c = 0; c < classes; ++c) {
            probs[r * classes + c] = std::exp(logits[r * classes + c] - row_max[r]) / row_sum[r];
        }
    }

    out_argmax.assign(batch, 0);
    for (int r = 0; r < batch; ++r) {
        float max_val = -FLT_MAX;
        int max_idx = 0;
        for (int c = 0; c < classes; ++c) {
            float v = probs[r * classes + c];
            if (v > max_val) {
                max_val = v;
                max_idx = c;
            }
        }
        out_argmax[r] = max_idx;
    }
}

int main() {
    const int batch = 128;
    const int input_stride = 160;
    const int seq_len = 128;
    const int dim = 64;
    const int hidden = 128;
    const int classes = 5;
    const int vocab = 103569;
    const int pad_token = 0;
    const int unk_id = 1;
    const float alpha = 0.01f;
    const float eps = 1e-5f;

    std::vector<int> h_lengths(batch, seq_len);
    for (int i = 0; i < batch; ++i) {
        h_lengths[i] = (i % seq_len) + 1;
    }

    std::vector<int> h_input(batch * input_stride, 0);
    for (int s = 0; s < batch; ++s) {
        for (int p = 0; p < input_stride; ++p) {
            h_input[s * input_stride + p] = (s * 131 + p * 7) % vocab;
        }
    }

    std::vector<float> h_embedding(vocab * dim, 0.0f);
    for (int v = 0; v < vocab; ++v) {
        for (int d = 0; d < dim; ++d) {
            h_embedding[v * dim + d] = 0.001f * static_cast<float>((v + d) % 97);
        }
    }

    std::vector<float> h_bias(dim, 0.0f);
    std::vector<float> h_gamma(dim, 1.0f);
    std::vector<float> h_beta(dim, 0.0f);
    for (int d = 0; d < dim; ++d) {
        h_bias[d] = 0.01f * static_cast<float>(d);
        h_gamma[d] = 1.0f;
        h_beta[d] = 0.0f;
    }

    std::vector<float> h_w_hidden(dim * hidden, 0.0f);
    for (int i = 0; i < dim * hidden; ++i) {
        h_w_hidden[i] = 0.002f * static_cast<float>(i % 89);
    }

    std::vector<float> h_w_logits(hidden * classes, 0.0f);
    for (int i = 0; i < hidden * classes; ++i) {
        h_w_logits[i] = 0.003f * static_cast<float>(i % 83);
    }

    std::vector<int> h_cpu_argmax;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_reference_pipeline(h_input, h_lengths, h_embedding, h_bias, h_gamma, h_beta,
                           h_w_hidden, h_w_logits, h_cpu_argmax, batch, input_stride,
                           seq_len, dim, hidden, classes, vocab, pad_token, unk_id,
                           alpha, eps);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    int* d_input = nullptr;
    int* d_lengths = nullptr;
    int* d_tokens = nullptr;
    float* d_embedding = nullptr;
    float* d_embed_out = nullptr;
    float* d_pos_out = nullptr;
    float* d_weights = nullptr;
    float* d_pooled = nullptr;
    float* d_bias = nullptr;
    float* d_biased = nullptr;
    float* d_activated = nullptr;
    float* d_mean = nullptr;
    float* d_var = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_bn_out = nullptr;
    float* d_w_hidden = nullptr;
    float* d_hidden_out = nullptr;
    float* d_w_logits = nullptr;
    float* d_logits = nullptr;
    float* d_probs = nullptr;
#if !USE_FUSED_SOFTMAX
    float* d_row_max = nullptr;
    float* d_row_sum = nullptr;
#endif
    int* d_argmax = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, h_lengths.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tokens, batch * seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_embedding, h_embedding.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_embed_out, batch * seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_out, batch * seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, batch * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pooled, batch * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_biased, batch * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_activated, batch * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, h_gamma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, h_beta.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bn_out, batch * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_hidden, h_w_hidden.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden_out, batch * hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_logits, h_w_logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, batch * classes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, batch * classes * sizeof(float)));
#if !USE_FUSED_SOFTMAX
    CUDA_CHECK(cudaMalloc(&d_row_max, batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_sum, batch * sizeof(float)));
#endif
    CUDA_CHECK(cudaMalloc(&d_argmax, batch * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), h_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_embedding, h_embedding.data(), h_embedding.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_hidden, h_w_hidden.data(), h_w_hidden.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_logits, h_w_logits.data(), h_w_logits.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_weights(batch * seq_len, 0.0f);
    for (int s = 0; s < batch; ++s) {
        for (int t = 0; t < seq_len; ++t) {
            h_weights[s * seq_len + t] = (t < h_lengths[s]) ? 1.0f : 0.0f;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
#if USE_CUBLAS_TF32
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

    // Warm up cuBLAS to avoid first-call overhead in timing.
#if USE_CUBLAS_TF32
    {
        const float alpha_gemm = 1.0f;
        const float beta_gemm = 0.0f;
        int m = hidden;
        int n = batch;
        int k = dim;
        int lda = hidden;
        int ldb = dim;
        int ldc = hidden;
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha_gemm,
            d_w_hidden,
            CUDA_R_32F,
            lda,
            d_bn_out,
            CUDA_R_32F,
            ldb,
            &beta_gemm,
            d_hidden_out,
            CUDA_R_32F,
            ldc,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    const int threads = 256;
    int total_tokens = batch * seq_len;
    int total_embed = total_tokens * dim;

    int blocks_pad = (batch * seq_len + threads - 1) / threads;
    pad_truncate_kernel<<<blocks_pad, threads>>>(d_input, d_lengths, d_tokens, batch, input_stride, seq_len, pad_token);

    dim3 block_embed(16, 8, 1);
    dim3 grid_embed(1, (total_tokens + block_embed.y - 1) / block_embed.y, 1);
    embedding_lookup_kernel<<<grid_embed, block_embed>>>(d_tokens, d_embedding, d_embed_out, total_tokens, dim, vocab, unk_id);

    int blocks_pos = (total_embed + threads - 1) / threads;
    positional_encoding_kernel<<<blocks_pos, threads>>>(d_embed_out, d_pos_out, total_tokens, dim);

    dim3 block_pool(seq_len, 1, 1);
    dim3 grid_pool(batch, dim, 1);
    size_t shared_pool = static_cast<size_t>(seq_len) * sizeof(float) * 2;
    weighted_mean_pooling_kernel<<<grid_pool, block_pool, shared_pool>>>(d_pos_out, d_weights, d_pooled, batch, seq_len, dim);

    int total_pooled = batch * dim;
    int blocks_bias = (total_pooled + threads - 1) / threads;
    int blocks_fused = ((total_pooled + 3) / 4 + threads - 1) / threads;
    bias_leaky_relu_kernel<<<blocks_fused, threads>>>(
        d_pooled,
        d_bias,
        d_activated,
        batch,
        dim,
        alpha);

    int blocks_bn = dim;
    batchnorm_mean_kernel<<<blocks_bn, threads>>>(d_activated, d_mean, batch, dim);
    batchnorm_var_kernel<<<blocks_bn, threads>>>(d_activated, d_mean, d_var, batch, dim);
    batchnorm_apply_kernel<<<blocks_bias, threads>>>(d_activated, d_mean, d_var, d_gamma, d_beta, d_bn_out, batch, dim, eps);

    // TF32 GEMM via cuBLAS: compute (A*B)^T as B^T * A^T (column-major view).
#if USE_CUBLAS_TF32
    {
        const float alpha_gemm = 1.0f;
        const float beta_gemm = 0.0f;
        int m = hidden;
        int n = batch;
        int k = dim;
        int lda = hidden; // B treated as [hidden x dim] column-major (B^T)
        int ldb = dim;    // A treated as [dim x batch] column-major (A^T)
        int ldc = hidden; // C treated as [hidden x batch] column-major (C^T)

        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha_gemm,
            d_w_hidden,
            CUDA_R_32F,
            lda,
            d_bn_out,
            CUDA_R_32F,
            ldb,
            &beta_gemm,
            d_hidden_out,
            CUDA_R_32F,
            ldc,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT));
    }
#else
    dim3 block_gemm(TILE, TILE, 1);
    dim3 grid_gemm((hidden + TILE - 1) / TILE, (batch + TILE - 1) / TILE, 1);
    gemm_tiled_kernel<<<grid_gemm, block_gemm>>>(d_bn_out, d_w_hidden, d_hidden_out, batch, dim, hidden);
#endif

    int total_logits = batch * classes;
    int blocks_logits = (total_logits + threads - 1) / threads;
    logit_projection_kernel<<<blocks_logits, threads>>>(d_hidden_out, d_w_logits, d_logits, batch, hidden, classes);

    int threads_softmax = 32;
#if USE_FUSED_SOFTMAX
    size_t shared_softmax = 2 * sizeof(float);
    softmax_fused_kernel<<<batch, threads_softmax, shared_softmax>>>(d_logits, d_probs, batch, classes);
#else
    softmax_row_max_kernel<<<batch, threads_softmax>>>(d_logits, d_row_max, batch, classes);
    softmax_row_sum_kernel<<<batch, threads_softmax>>>(d_logits, d_row_max, d_row_sum, batch, classes);
    softmax_normalize_kernel<<<blocks_logits, threads>>>(d_logits, d_row_max, d_row_sum, d_probs, batch, classes);
#endif

    argmax_kernel<<<batch, threads_softmax>>>(d_probs, d_argmax, batch, classes);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    std::vector<int> h_gpu_argmax(batch, 0);
    CUDA_CHECK(cudaMemcpy(h_gpu_argmax.data(), d_argmax, batch * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < batch; ++i) {
        if (h_gpu_argmax[i] != h_cpu_argmax[i]) {
            std::cerr << "Mismatch at row " << i << ": got " << h_gpu_argmax[i]
                      << ", expected " << h_cpu_argmax[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Pipeline baseline: PASS\n";
    } else {
        std::cout << "Pipeline baseline: FAIL\n";
    }

    std::cout << "CPU time (ms): " << cpu_ms << "\n";
    std::cout << "GPU time (ms): " << gpu_ms << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
#if USE_CUBLAS_TF32
    CUBLAS_CHECK(cublasDestroy(handle));
#endif

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_embedding));
    CUDA_CHECK(cudaFree(d_embed_out));
    CUDA_CHECK(cudaFree(d_pos_out));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_pooled));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_biased));
    CUDA_CHECK(cudaFree(d_activated));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_var));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_bn_out));
    CUDA_CHECK(cudaFree(d_w_hidden));
    CUDA_CHECK(cudaFree(d_hidden_out));
    CUDA_CHECK(cudaFree(d_w_logits));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_probs));
#if !USE_FUSED_SOFTMAX
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));
#endif
    CUDA_CHECK(cudaFree(d_argmax));

    return ok ? 0 : 1;
}
