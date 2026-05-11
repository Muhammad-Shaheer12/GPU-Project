#include <torch/extension.h>
#include <vector>
#include "bridge.hpp"

// ============================================================
// Helper: ensure tensor is on CUDA, contiguous, and correct type
// ============================================================
static torch::Tensor ensure_float_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    if (t.scalar_type() != torch::kFloat32) t = t.to(torch::kFloat32);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t;
}

static torch::Tensor ensure_int_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    if (t.scalar_type() == torch::kLong) t = t.toType(torch::kInt32);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t;
}

// ============================================================
// PyTorch Bindings Implementation
// ============================================================

torch::Tensor pad_truncate(torch::Tensor input_tokens, torch::Tensor input_lengths, int fixed_len, int pad_token) {
    input_tokens = ensure_int_cuda(input_tokens, "input_tokens");
    input_lengths = ensure_int_cuda(input_lengths, "input_lengths");
    int batch = input_tokens.size(0);
    auto output_tokens = torch::empty({batch, fixed_len}, torch::TensorOptions().dtype(torch::kInt32).device(input_tokens.device()));
    launch_pad_truncate(input_tokens.data_ptr<int>(), input_lengths.data_ptr<int>(), output_tokens.data_ptr<int>(), batch, input_tokens.size(1), fixed_len, pad_token);
    return output_tokens;
}

torch::Tensor embedding_lookup(torch::Tensor tokens, torch::Tensor embedding_table, int unk_id) {
    tokens = ensure_int_cuda(tokens, "tokens");
    embedding_table = ensure_float_cuda(embedding_table, "embedding_table");
    auto output = torch::empty({tokens.numel(), (int)embedding_table.size(1)}, torch::TensorOptions().device(tokens.device()));
    launch_embedding_lookup(tokens.data_ptr<int>(), embedding_table.data_ptr<float>(), output.data_ptr<float>(), tokens.numel(), embedding_table.size(1), embedding_table.size(0), unk_id);
    return output;
}

torch::Tensor positional_encoding(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    auto output = torch::empty_like(input);
    launch_positional_encoding(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    return output;
}

torch::Tensor weighted_mean_pooling(torch::Tensor input, torch::Tensor weights) {
    input = ensure_float_cuda(input, "input");
    weights = ensure_float_cuda(weights, "weights");
    auto output = torch::empty({input.size(0), input.size(2)}, torch::TensorOptions().device(input.device()));
    launch_weighted_mean_pooling(input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1), input.size(2));
    return output;
}

torch::Tensor bias_add(torch::Tensor input, torch::Tensor bias) {
    input = ensure_float_cuda(input, "input");
    bias = ensure_float_cuda(bias, "bias");
    auto output = torch::empty_like(input);
    launch_bias_add(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    return output;
}

torch::Tensor leaky_relu(torch::Tensor input, float alpha) {
    input = ensure_float_cuda(input, "input");
    auto output = torch::empty_like(input);
    launch_leaky_relu(input.data_ptr<float>(), output.data_ptr<float>(), input.numel(), alpha);
    return output;
}

torch::Tensor batchnorm_mean(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    auto mean = torch::empty({input.size(1)}, torch::TensorOptions().device(input.device()));
    launch_batchnorm_mean(input.data_ptr<float>(), mean.data_ptr<float>(), input.size(0), input.size(1));
    return mean;
}

torch::Tensor batchnorm_var(torch::Tensor input, torch::Tensor mean) {
    input = ensure_float_cuda(input, "input");
    mean = ensure_float_cuda(mean, "mean");
    auto var = torch::empty({input.size(1)}, torch::TensorOptions().device(input.device()));
    launch_batchnorm_var(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), input.size(0), input.size(1));
    return var;
}

torch::Tensor batchnorm_apply(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps) {
    input = ensure_float_cuda(input, "input");
    auto output = torch::empty_like(input);
    launch_batchnorm_apply(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1), eps);
    return output;
}

torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B) {
    A = ensure_float_cuda(A, "A");
    B = ensure_float_cuda(B, "B");
    auto C = torch::empty({A.size(0), B.size(1)}, torch::TensorOptions().device(A.device()));
    launch_gemm_tiled(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), A.size(0), A.size(1), B.size(1));
    return C;
}

torch::Tensor logit_projection(torch::Tensor input, torch::Tensor weights) {
    input = ensure_float_cuda(input, "input");
    weights = ensure_float_cuda(weights, "weights");
    auto output = torch::empty({input.size(0), weights.size(1)}, torch::TensorOptions().device(input.device()));
    launch_logit_projection(input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1), weights.size(1));
    return output;
}

torch::Tensor softmax_row_max(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    auto row_max = torch::empty({input.size(0)}, torch::TensorOptions().device(input.device()));
    launch_softmax_row_max(input.data_ptr<float>(), row_max.data_ptr<float>(), input.size(0), input.size(1));
    return row_max;
}

torch::Tensor softmax_row_sum(torch::Tensor input, torch::Tensor row_max) {
    input = ensure_float_cuda(input, "input");
    row_max = ensure_float_cuda(row_max, "row_max");
    auto row_sum = torch::empty({input.size(0)}, torch::TensorOptions().device(input.device()));
    launch_softmax_row_sum(input.data_ptr<float>(), row_max.data_ptr<float>(), row_sum.data_ptr<float>(), input.size(0), input.size(1));
    return row_sum;
}

torch::Tensor softmax_normalize(torch::Tensor input, torch::Tensor row_max, torch::Tensor row_sum) {
    input = ensure_float_cuda(input, "input");
    auto output = torch::empty_like(input);
    launch_softmax_normalize(input.data_ptr<float>(), row_max.data_ptr<float>(), row_sum.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    return output;
}

torch::Tensor argmax(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    auto output = torch::empty({input.size(0)}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    launch_argmax(input.data_ptr<float>(), output.data_ptr<int>(), input.size(0), input.size(1));
    return output;
}

torch::Tensor fused_bias_leaky_relu(torch::Tensor input, torch::Tensor bias, float alpha) {
    input = ensure_float_cuda(input, "input");
    bias = ensure_float_cuda(bias, "bias");
    auto output = torch::empty_like(input);
    launch_fused_bias_leaky_relu(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1), alpha);
    return output;
}

torch::Tensor fused_softmax(torch::Tensor input) {
    input = ensure_float_cuda(input, "input");
    auto output = torch::empty_like(input);
    launch_fused_softmax(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pad_truncate", &pad_truncate);
    m.def("embedding_lookup", &embedding_lookup);
    m.def("positional_encoding", &positional_encoding);
    m.def("weighted_mean_pooling", &weighted_mean_pooling);
    m.def("bias_add", &bias_add);
    m.def("leaky_relu", &leaky_relu);
    m.def("batchnorm_mean", &batchnorm_mean);
    m.def("batchnorm_var", &batchnorm_var);
    m.def("batchnorm_apply", &batchnorm_apply);
    m.def("gemm_tiled", &gemm_tiled);
    m.def("logit_projection", &logit_projection);
    m.def("softmax_row_max", &softmax_row_max);
    m.def("softmax_row_sum", &softmax_row_sum);
    m.def("softmax_normalize", &softmax_normalize);
    m.def("argmax", &argmax);
    m.def("fused_bias_leaky_relu", &fused_bias_leaky_relu);
    m.def("fused_softmax", &fused_softmax);
}
