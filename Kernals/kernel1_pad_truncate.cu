#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Minimal CUDA error checking for fast feedback during kernel development.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

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

int main() {
    // Small synthetic test to validate padding and truncation behavior.
    const int batch = 3;
    const int input_stride = 160;
    const int fixed_len = 128;
    const int pad_token = 0;

    std::vector<int> h_lengths = {3, 128, 140};
    std::vector<int> h_input(batch * input_stride, -1);

    // Fill each sentence with a distinct pattern: s*1000 + position.
    for (int s = 0; s < batch; ++s) {
        for (int p = 0; p < input_stride; ++p) {
            h_input[s * input_stride + p] = s * 1000 + p;
        }
    }

    std::vector<int> h_output(batch * fixed_len, -1);
    std::vector<int> h_expected(batch * fixed_len, -1);

    // Build CPU reference output for correctness checking.
    for (int s = 0; s < batch; ++s) {
        int len = h_lengths[s];
        int capped = len < fixed_len ? len : fixed_len;
        for (int p = 0; p < fixed_len; ++p) {
            if (p < capped) {
                h_expected[s * fixed_len + p] = h_input[s * input_stride + p];
            } else {
                h_expected[s * fixed_len + p] = pad_token;
            }
        }
    }

    int* d_input = nullptr;
    int* d_lengths = nullptr;
    int* d_output = nullptr;

    // Device allocations and host-to-device copies.
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, h_lengths.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), h_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Launch one thread per output element.
    const int threads = 256;
    const int total = batch * fixed_len;
    const int blocks = (total + threads - 1) / threads;

    pad_truncate_kernel<<<blocks, threads>>>(
        d_input,
        d_lengths,
        d_output,
        batch,
        input_stride,
        fixed_len,
        pad_token);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify GPU output matches the CPU reference.
    bool ok = true;
    for (size_t i = 0; i < h_output.size(); ++i) {
        if (h_output[i] != h_expected[i]) {
            std::cerr << "Mismatch at " << i << ": got " << h_output[i]
                      << ", expected " << h_expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Kernel 1 pad/truncate: PASS\n";
    } else {
        std::cout << "Kernel 1 pad/truncate: FAIL\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}
