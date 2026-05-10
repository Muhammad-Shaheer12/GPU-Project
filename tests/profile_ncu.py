import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_ext')))
import custom_cuda_ops

def main():
    batch = 2048
    input_stride = 160
    fixed_len = 128
    pad_token = 0

    input_tokens = torch.randint(10, 1000, (batch, input_stride), dtype=torch.int32).cuda()
    input_lengths = torch.randint(10, 150, (batch,), dtype=torch.int32).cuda()

    # Launch kernel EXACTLY ONCE so ncu can profile it quickly
    out = custom_cuda_ops.pad_truncate(input_tokens, input_lengths, fixed_len, pad_token)

if __name__ == "__main__":
    main()
