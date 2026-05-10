#include <torch/extension.h>
#include <vector>

void launch_pad_truncate(const int* input_tokens,
                         const int* input_lengths,
                         int* output_tokens,
                         int batch,
                         int input_stride,
                         int fixed_len,
                         int pad_token);

// C++ wrapper for the kernel
torch::Tensor pad_truncate(torch::Tensor input_tokens,
                           torch::Tensor input_lengths,
                           int fixed_len,
                           int pad_token) {
    // Ensure inputs are contiguous and on CUDA
    TORCH_CHECK(input_tokens.is_cuda(), "input_tokens must be a CUDA tensor");
    TORCH_CHECK(input_lengths.is_cuda(), "input_lengths must be a CUDA tensor");
    TORCH_CHECK(input_tokens.is_contiguous(), "input_tokens must be contiguous");
    TORCH_CHECK(input_lengths.is_contiguous(), "input_lengths must be contiguous");
    
    // Convert to int32 if they are int64 since kernel uses int*
    if (input_tokens.scalar_type() == torch::kLong) {
        input_tokens = input_tokens.toType(torch::kInt32);
    }
    if (input_lengths.scalar_type() == torch::kLong) {
        input_lengths = input_lengths.toType(torch::kInt32);
    }

    int batch = input_tokens.size(0);
    int input_stride = input_tokens.size(1);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_tokens.device());
    torch::Tensor output_tokens = torch::empty({batch, fixed_len}, options);

    launch_pad_truncate(
        input_tokens.data_ptr<int>(),
        input_lengths.data_ptr<int>(),
        output_tokens.data_ptr<int>(),
        batch,
        input_stride,
        fixed_len,
        pad_token
    );

    return output_tokens;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pad_truncate", &pad_truncate, "Pad or truncate tokens (CUDA)");
}

