#include <cuda_runtime.h>

__global__ void pad_truncate_kernel(const int* input_tokens,
                                    const int* input_lengths,
                                    int* output_tokens,
                                    int batch,
                                    int input_stride,
                                    int fixed_len,
                                    int pad_token);

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
        input_tokens,
        input_lengths,
        output_tokens,
        batch,
        input_stride,
        fixed_len,
        pad_token
    );
}
