#include <torch/extension.h>
#include <vector>

// ============================================================
// Forward declarations of launch wrappers (defined in custom_kernels_wrapper.cu)
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

void launch_gemm_tiled(const float* A, const float* B, float* C,
                        int M, int K, int N);

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
void launch_fused_bias_leaky_relu(const float* input, const float* bias, float* output, int rows, int cols, float alpha);
void launch_fused_softmax(const float* input, float* output, int batch, int classes);


// ============================================================
// Helper: ensure tensor is float32, contiguous, on CUDA
// ============================================================
static torch::Tensor ensure_float_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    if (t.scalar_type() != torch::kFloat32) {
        t = t.to(torch::kFloat32);
    }
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t;
}

static torch::Tensor ensure_int_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    if (t.scalar_type() == torch::kLong) {
        t = t.toType(torch::kInt32);
    }
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t;
}


// ============================================================
// K1: pad_truncate
// ============================================================
torch::Tensor pad_truncate(torch::Tensor input_tokens,
                            torch::Tensor input_lengths,
                            int fixed_len,
                            int pad_token) {
    input_tokens = ensure_int_cuda(input_tokens, "input_tokens");
    input_lengths = ensure_int_cuda(input_lengths, "input_lengths");

    int batch = input_tokens.size(0);
    int input_stride = input_tokens.size(1);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_tokens.device());
    torch::Tensor output_tokens = torch::empty({batch, fixed_len}, options);

    launch_pad_truncate(input_tokens.data_ptr<int>(),
                        input_lengths.data_ptr<int>(),
                        output_tokens.data_ptr<int>(),
                        batch, input_stride, fixed_len, pad_token);
    return output_tokens;
}

// ============================================================
// K2: embedding_lookup
// ============================================================
torch::Tensor embedding_lookup(torch::Tensor tokens,
                                torch::Tensor embedding_table,
                                int unk_id) {
    tokens = ensure_int_cuda(tokens, "tokens");
    embedding_table = ensure_float_cuda(embedding_table, "embedding_table");

    int total_tokens = tokens.numel();
    int vocab = embedding_table.size(0);
    int dim = embedding_table.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tokens.device());
    torch::Tensor output = torch::empty({total_tokens, dim}, options);

    launch_embedding_lookup(tokens.data_ptr<int>(),
                             embedding_table.data_ptr<float>(),
                             output.data_ptr<float>(),
                             total_tokens, dim, vocab, unk_id);
    return output;
}

// ============================================================
// K3: positional_encoding
// ============================================================
torch::Tensor positional_encoding(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [total_tokens x dim]");

    int total_tokens = input.size(0);
    int dim = input.size(1);

    auto output = torch::empty_like(input);

    launch_positional_encoding(input.data_ptr<float>(),
                                output.data_ptr<float>(),
                                total_tokens, dim);
    return output;
}

// ============================================================
// K4: weighted_mean_pooling
// ============================================================
torch::Tensor weighted_mean_pooling(torch::Tensor input,
                                     torch::Tensor weights) {
    input = ensure_float_cuda(input, "input");
    weights = ensure_float_cuda(weights, "weights");

    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch x seq_len x dim]");
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);

    TORCH_CHECK(weights.numel() == batch * seq_len,
                "weights must have batch*seq_len elements");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor output = torch::empty({batch, dim}, options);

    launch_weighted_mean_pooling(input.data_ptr<float>(),
                                  weights.data_ptr<float>(),
                                  output.data_ptr<float>(),
                                  batch, seq_len, dim);
    return output;
}

// ============================================================
// K5: bias_add
// ============================================================
torch::Tensor bias_add(torch::Tensor input, torch::Tensor bias) {
    input = ensure_float_cuda(input, "input");
    bias = ensure_float_cuda(bias, "bias");

    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty_like(input);

    launch_bias_add(input.data_ptr<float>(),
                     bias.data_ptr<float>(),
                     output.data_ptr<float>(),
                     rows, cols);
    return output;
}

// ============================================================
// K6: leaky_relu
// ============================================================
torch::Tensor leaky_relu(torch::Tensor input, float alpha) {
    input = ensure_float_cuda(input, "input");

    int total = input.numel();
    auto output = torch::empty_like(input);

    launch_leaky_relu(input.data_ptr<float>(),
                       output.data_ptr<float>(),
                       total, alpha);
    return output;
}

// ============================================================
// K7: batchnorm_mean
// ============================================================
torch::Tensor batchnorm_mean(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch x features]");

    int batch = input.size(0);
    int features = input.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor mean = torch::empty({features}, options);

    launch_batchnorm_mean(input.data_ptr<float>(),
                           mean.data_ptr<float>(),
                           batch, features);
    return mean;
}

// ============================================================
// K8: batchnorm_var
// ============================================================
torch::Tensor batchnorm_var(torch::Tensor input, torch::Tensor mean) {
    input = ensure_float_cuda(input, "input");
    mean = ensure_float_cuda(mean, "mean");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch x features]");

    int batch = input.size(0);
    int features = input.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor var = torch::empty({features}, options);

    launch_batchnorm_var(input.data_ptr<float>(),
                          mean.data_ptr<float>(),
                          var.data_ptr<float>(),
                          batch, features);
    return var;
}

// ============================================================
// K9: batchnorm_apply
// ============================================================
torch::Tensor batchnorm_apply(torch::Tensor input,
                               torch::Tensor mean,
                               torch::Tensor var,
                               torch::Tensor gamma,
                               torch::Tensor beta,
                               float eps) {
    input = ensure_float_cuda(input, "input");
    mean = ensure_float_cuda(mean, "mean");
    var = ensure_float_cuda(var, "var");
    gamma = ensure_float_cuda(gamma, "gamma");
    beta = ensure_float_cuda(beta, "beta");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch x features]");

    int batch = input.size(0);
    int features = input.size(1);

    auto output = torch::empty_like(input);

    launch_batchnorm_apply(input.data_ptr<float>(),
                            mean.data_ptr<float>(),
                            var.data_ptr<float>(),
                            gamma.data_ptr<float>(),
                            beta.data_ptr<float>(),
                            output.data_ptr<float>(),
                            batch, features, eps);
    return output;
}

// ============================================================
// K10: gemm_tiled
// ============================================================
torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B) {
    A = ensure_float_cuda(A, "A");
    B = ensure_float_cuda(B, "B");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    launch_gemm_tiled(A.data_ptr<float>(),
                       B.data_ptr<float>(),
                       C.data_ptr<float>(),
                       M, K, N);
    return C;
}

// ============================================================
// K11: logit_projection
// ============================================================
torch::Tensor logit_projection(torch::Tensor input,
                                torch::Tensor weights) {
    input = ensure_float_cuda(input, "input");
    weights = ensure_float_cuda(weights, "weights");
    TORCH_CHECK(input.dim() == 2 && weights.dim() == 2, "input and weights must be 2D");

    int batch = input.size(0);
    int hidden = input.size(1);
    int classes = weights.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor output = torch::empty({batch, classes}, options);

    launch_logit_projection(input.data_ptr<float>(),
                              weights.data_ptr<float>(),
                              output.data_ptr<float>(),
                              batch, hidden, classes);
    return output;
}

// ============================================================
// K12: softmax_row_max
// ============================================================
torch::Tensor softmax_row_max(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch x classes]");

    int batch = input.size(0);
    int classes = input.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor row_max = torch::empty({batch}, options);

    launch_softmax_row_max(input.data_ptr<float>(),
                             row_max.data_ptr<float>(),
                             batch, classes);
    return row_max;
}

// ============================================================
// K13: softmax_row_sum
// ============================================================
torch::Tensor softmax_row_sum(torch::Tensor input, torch::Tensor row_max) {
    input = ensure_float_cuda(input, "input");
    row_max = ensure_float_cuda(row_max, "row_max");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch x classes]");

    int batch = input.size(0);
    int classes = input.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor row_sum = torch::empty({batch}, options);

    launch_softmax_row_sum(input.data_ptr<float>(),
                             row_max.data_ptr<float>(),
                             row_sum.data_ptr<float>(),
                             batch, classes);
    return row_sum;
}

// ============================================================
// K14: softmax_normalize
// ============================================================
torch::Tensor softmax_normalize(torch::Tensor input,
                                 torch::Tensor row_max,
                                 torch::Tensor row_sum) {
    input = ensure_float_cuda(input, "input");
    row_max = ensure_float_cuda(row_max, "row_max");
    row_sum = ensure_float_cuda(row_sum, "row_sum");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch x classes]");

    int batch = input.size(0);
    int classes = input.size(1);

    auto output = torch::empty_like(input);

    launch_softmax_normalize(input.data_ptr<float>(),
                               row_max.data_ptr<float>(),
                               row_sum.data_ptr<float>(),
                               output.data_ptr<float>(),
                               batch, classes);
    return output;
}

// ============================================================
// K15: argmax
// ============================================================
torch::Tensor argmax(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch x classes]");

    int batch = input.size(0);
    int classes = input.size(1);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor output = torch::empty({batch}, options);

    launch_argmax(input.data_ptr<float>(),
                   output.data_ptr<int>(),
                   batch, classes);
    return output;
}

// ============================================================
// K16: fused_bias_leaky_relu
// ============================================================
torch::Tensor fused_bias_leaky_relu(torch::Tensor input,
                                     torch::Tensor bias,
                                     float alpha) {
    input = ensure_float_cuda(input, "input");
    bias = ensure_float_cuda(bias, "bias");
    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);
    launch_fused_bias_leaky_relu(input.data_ptr<float>(),
                                  bias.data_ptr<float>(),
                                  output.data_ptr<float>(),
                                  rows, cols, alpha);
    return output;
}

// ============================================================
// K17: fused_softmax
// ============================================================
torch::Tensor fused_softmax(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    int batch = input.size(0);
    int classes = input.size(1);
    auto output = torch::empty_like(input);
    launch_fused_softmax(input.data_ptr<float>(),
                          output.data_ptr<float>(),
                          batch, classes);
    return output;
}


// ============================================================
// PyBind11 module registration
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pad_truncate", &pad_truncate, "K1: Pad/truncate token sequences");
    m.def("embedding_lookup", &embedding_lookup, "K2: Embedding table lookup");
    m.def("positional_encoding", &positional_encoding, "K3: Sinusoidal positional encoding");
    m.def("weighted_mean_pooling", &weighted_mean_pooling, "K4: Weighted mean pooling over sequence");
    m.def("bias_add", &bias_add, "K5: Add bias vector to rows");
    m.def("leaky_relu", &leaky_relu, "K6: Leaky ReLU activation");
    m.def("batchnorm_mean", &batchnorm_mean, "K7: BatchNorm mean per feature");
    m.def("batchnorm_var", &batchnorm_var, "K8: BatchNorm variance per feature");
    m.def("batchnorm_apply", &batchnorm_apply, "K9: BatchNorm apply (normalize + scale + shift)");
    m.def("gemm_tiled", &gemm_tiled, "K10: Tiled GEMM (C = A * B)");
    m.def("logit_projection", &logit_projection, "K11: Logit projection (batch matmul)");
    m.def("softmax_row_max", &softmax_row_max, "K12: Softmax row-wise max");
    m.def("softmax_row_sum", &softmax_row_sum, "K13: Softmax row-wise sum of exp");
    m.def("softmax_normalize", &softmax_normalize, "K14: Softmax normalize");
    m.def("argmax", &argmax, "K15: Argmax per row");
    m.def("fused_bias_leaky_relu", &fused_bias_leaky_relu, "K16: Fused bias + leaky ReLU");
    m.def("fused_softmax", &fused_softmax, "K17: Fused single-pass softmax");
}
