#include <cuda_runtime.h>

// Kernel 1: pad or truncate each sentence to a fixed length.
__global__ void pad_truncate_kernel(const int* input_tokens,
                                    const int* input_lengths,
                                    int* output_tokens,
                                    int batch,
                                    int input_stride,
                                    int fixed_len,
                                    int pad_token) {
    // One thread handles one output element (sentence, position).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * fixed_len;
    if (idx >= total) {
        return;
    }

    // Map the flat index to sentence id and position inside that sentence.
    int sentence = idx / fixed_len;
    int pos = idx - sentence * fixed_len;
    int len = input_lengths[sentence];

    // Copy real tokens up to length; otherwise pad.
    if (pos < len) {
        output_tokens[sentence * fixed_len + pos] =
            input_tokens[sentence * input_stride + pos];
    } else {
        output_tokens[sentence * fixed_len + pos] = pad_token;
    }
}
