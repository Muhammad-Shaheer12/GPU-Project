import torch
import torch.utils.benchmark as benchmark
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_ext')))
import custom_cuda_ops

def pytorch_pad_truncate(input_tokens, input_lengths, fixed_len, pad_token):
    batch = input_tokens.shape[0]
    output_tokens = torch.full((batch, fixed_len), pad_token, dtype=input_tokens.dtype, device=input_tokens.device)
    for i in range(batch):
        l = min(input_lengths[i].item(), fixed_len)
        output_tokens[i, :l] = input_tokens[i, :l]
    return output_tokens

def main():
    batch = 2048 # Using a large realistic batch size for performance testing
    input_stride = 160
    fixed_len = 128
    pad_token = 0

    input_tokens = torch.randint(10, 1000, (batch, input_stride), dtype=torch.int32).cuda()
    input_lengths = torch.randint(10, 150, (batch,), dtype=torch.int32).cuda()

    # 1. Benchmark PyTorch
    t0 = benchmark.Timer(
        stmt='pytorch_pad_truncate(input_tokens, input_lengths, fixed_len, pad_token)',
        setup='from __main__ import pytorch_pad_truncate',
        globals={'input_tokens': input_tokens, 'input_lengths': input_lengths, 'fixed_len': fixed_len, 'pad_token': pad_token}
    )

    # 2. Benchmark Custom CUDA
    t1 = benchmark.Timer(
        stmt='custom_cuda_ops.pad_truncate(input_tokens, input_lengths, fixed_len, pad_token)',
        setup='import custom_cuda_ops',
        globals={'input_tokens': input_tokens, 'input_lengths': input_lengths, 'fixed_len': fixed_len, 'pad_token': pad_token}
    )

    print("--- Execution Time Comparison (Batch Size 2048) ---")
    print("Normal PyTorch Implementation:")
    print(t0.timeit(100))
    
    print("Custom CUDA Kernel Implementation:")
    print(t1.timeit(100))

if __name__ == "__main__":
    main()
