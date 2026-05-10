#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <chrono>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

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

__global__ void pad_truncate_kernel(const int* input_tokens,
                                    const int* input_lengths,
                                    int* output_tokens,
                                    int batch,
                                    int input_stride,
                                    int fixed_len,
                                    int pad_token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * fixed_len;
    if (idx >= total) {
        return;
    }

    int sentence = idx / fixed_len;
    int pos = idx - sentence * fixed_len;
    int len = input_lengths[sentence];

    if (pos < len) {
        output_tokens[sentence * fixed_len + pos] =
            input_tokens[sentence * input_stride + pos];
    } else {
        output_tokens[sentence * fixed_len + pos] = pad_token;
    }
}

__global__ void embedding_lookup_kernel(const int* __restrict__ tokens,
                                        const float* __restrict__ embedding,
                                        float* __restrict__ output,
                                        int total_tokens,
                                        int dim,
                                        int vocab,
                                        int unk_id) {
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int base_dim = lane * 4;

    if (token_idx >= total_tokens || base_dim >= dim) {
        return;
    }

    int token = tokens[token_idx];
    if (token < 0 || token >= vocab) {
        token = unk_id;
    }

    int emb_offset = token * dim + base_dim;
    int out_offset = token_idx * dim + base_dim;

    if (base_dim + 3 < dim) {
        float4 v = reinterpret_cast<const float4*>(embedding + emb_offset)[0];
        reinterpret_cast<float4*>(output + out_offset)[0] = v;
    } else {
        for (int k = 0; k < 4 && base_dim + k < dim; ++k) {
            output[out_offset + k] = embedding[emb_offset + k];
        }
    }
}

__global__ void positional_encoding_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int total_tokens,
                                           int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * dim;
    if (idx >= total) {
        return;
    }

    int token = idx / dim;
    int d = idx - token * dim;

    int pair = d / 2;
    float exponent = (2.0f * pair) / static_cast<float>(dim);
    float denom = expf(logf(10000.0f) * exponent);
    float angle = static_cast<float>(token) / denom;

    float pe = (d % 2 == 0) ? sinf(angle) : cosf(angle);
    output[idx] = input[idx] + pe;
}

__global__ void weighted_mean_pooling_kernel(const float* __restrict__ input,
                                             const float* __restrict__ weights,
                                             float* __restrict__ output,
                                             int batch,
                                             int seq_len,
                                             int dim) {
    int sentence = blockIdx.x;
    int d = blockIdx.y;
    int t = threadIdx.x;

    if (sentence >= batch || d >= dim || t >= seq_len) {
        return;
    }

    int token_idx = sentence * seq_len + t;
    float w = weights[token_idx];
    float v = input[token_idx * dim + d];

    extern __shared__ float shared[];
    float* shared_val = shared;
    float* shared_w = shared + seq_len;

    shared_val[t] = v * w;
    shared_w[t] = w;
    __syncthreads();

    for (int stride = seq_len / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            shared_val[t] += shared_val[t + stride];
            shared_w[t] += shared_w[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        float denom = shared_w[0];
        output[sentence * dim + d] = (denom > 0.0f) ? (shared_val[0] / denom) : 0.0f;
    }
}

__global__ void bias_add_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int rows,
                                int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }

    int col = idx % cols;
    output[idx] = input[idx] + bias[col];
}

__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int total,
                                  float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    float x = input[idx];
    output[idx] = (x >= 0.0f) ? x : (alpha * x);
}

// Fused bias + leaky ReLU to reduce global memory traffic.
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

        float4 out;
        out.x = (x0 >= 0.0f) ? x0 : (alpha * x0);
        out.y = (x1 >= 0.0f) ? x1 : (alpha * x1);
        out.z = (x2 >= 0.0f) ? x2 : (alpha * x2);
        out.w = (x3 >= 0.0f) ? x3 : (alpha * x3);

        reinterpret_cast<float4*>(output + base)[0] = out;
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

__device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    }
    return v;
}

__device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

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

    float warp_sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

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

__global__ void batchnorm_var_kernel(const float* __restrict__ input,
                                     const float* __restrict__ mean,
                                     float* __restrict__ var,
                                     int batch,
                                     int features) {
    int feature = blockIdx.x;
    int tid = threadIdx.x;

    if (feature >= features) {
        return;
    }

    float sum = 0.0f;
    float m = mean[feature];
    for (int i = tid; i < batch; i += blockDim.x) {
        float diff = input[i * features + feature] - m;
        sum += diff * diff;
    }

    float warp_sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            var[feature] = block_sum / static_cast<float>(batch);
        }
    }
}

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

        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void logit_projection_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weights,
                                        float* __restrict__ output,
                                        int batch,
                                        int hidden,
                                        int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * classes;
    if (idx >= total) {
        return;
    }

    int row = idx / classes;
    int cls = idx - row * classes;

    float sum = 0.0f;
    const float* in_row = input + row * hidden;
    const float* w_col = weights + cls;

    for (int k = 0; k < hidden; ++k) {
        sum += in_row[k] * w_col[k * classes];
    }

    output[idx] = sum;
}

__global__ void softmax_row_max_kernel(const float* __restrict__ input,
                                       float* __restrict__ row_max,
                                       int batch,
                                       int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    float local_max = -FLT_MAX;
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        local_max = fmaxf(local_max, v);
    }

    float max_val = warp_reduce_max(local_max);
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
    if (tid == 0) {
        row_max[row] = max_val;
    }
}

__global__ void softmax_row_sum_kernel(const float* __restrict__ input,
                                       const float* __restrict__ row_max,
                                       float* __restrict__ row_sum,
                                       int batch,
                                       int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    float max_val = row_max[row];
    float local_sum = 0.0f;
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        local_sum += expf(v - max_val);
    }

    float sum_val = warp_reduce_sum(local_sum);
    sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);
    if (tid == 0) {
        row_sum[row] = sum_val;
    }
}

__global__ void softmax_normalize_kernel(const float* __restrict__ input,
                                         const float* __restrict__ row_max,
                                         const float* __restrict__ row_sum,
                                         float* __restrict__ output,
                                         int batch,
                                         int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * classes;
    if (idx >= total) {
        return;
    }

    int row = idx / classes;
    float max_val = row_max[row];
    float denom = row_sum[row];
    float v = input[idx];

    output[idx] = expf(v - max_val) / denom;
}

__global__ void softmax_fused_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int batch,
                                     int classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    extern __shared__ float sdata[];
    if (tid == 0) {
        float max_val = input[row * classes];
        for (int c = 1; c < classes; ++c) {
            float v = input[row * classes + c];
            if (v > max_val) {
                max_val = v;
            }
        }

        float sum_val = 0.0f;
        for (int c = 0; c < classes; ++c) {
            sum_val += expf(input[row * classes + c] - max_val);
        }

        sdata[0] = max_val;
        sdata[1] = sum_val;
    }
    __syncthreads();

    float max_val = sdata[0];
    float sum_val = sdata[1];
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = input[row * classes + c];
        output[row * classes + c] = expf(v - max_val) / sum_val;
    }
}
__device__ void warp_reduce_argmax(float& v, int& idx) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_v = __shfl_down_sync(0xFFFFFFFF, v, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_v > v) {
            v = other_v;
            idx = other_idx;
        }
    }
}

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

    warp_reduce_argmax(local_max, local_idx);
    if (tid == 0) {
        output[row] = local_idx;
    }
}

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
