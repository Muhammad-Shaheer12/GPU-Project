import torch
import custom_cuda_ops

batch = 3
input_stride = 160
fixed_len = 128
pad_token = 0

input_tokens = torch.full((batch, input_stride), -1, dtype=torch.int32).cuda()
for s in range(batch):
    for p in range(input_stride):
        input_tokens[s, p] = s * 1000 + p

input_lengths = torch.tensor([3, 128, 140], dtype=torch.int32).cuda()

print("Running custom pad_truncate kernel...")
output_tokens = custom_cuda_ops.pad_truncate(input_tokens, input_lengths, fixed_len, pad_token)

print("Output shape:", output_tokens.shape)
print("Output device:", output_tokens.device)
print("Sample of first sentence (length 3, should have 3 tokens and rest padded):")
print(output_tokens[0, :10])

print("\nSUCCESS! Custom Kernel runs directly from PyTorch.")
