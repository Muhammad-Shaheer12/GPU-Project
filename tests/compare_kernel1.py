import torch
import sys
import os

# Add the custom_ext path just in case, though it's installed globally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_ext')))
import custom_cuda_ops

def pytorch_pad_truncate(input_tokens, input_lengths, fixed_len, pad_token):
    """
    The 'normal' PyTorch equivalent of your custom kernel.
    It takes an existing 2D tensor, truncates rows that are too long,
    and pads rows that are too short up to fixed_len.
    """
    batch = input_tokens.shape[0]
    output_tokens = torch.full((batch, fixed_len), pad_token, dtype=input_tokens.dtype, device=input_tokens.device)
    
    for i in range(batch):
        l = min(input_lengths[i].item(), fixed_len)
        output_tokens[i, :l] = input_tokens[i, :l]
        
    return output_tokens

def main():
    batch = 5
    input_stride = 160
    fixed_len = 128
    pad_token = 0

    print(f"Creating test batch of {batch} sentences...")
    input_tokens = torch.randint(10, 1000, (batch, input_stride), dtype=torch.int32).cuda()
    
    # Randomly assign lengths (some shorter than 128 to test padding, some longer to test truncation)
    input_lengths = torch.tensor([50, 128, 150, 10, 140], dtype=torch.int32).cuda()

    print("\n1. Running Normal PyTorch Implementation...")
    pytorch_out = pytorch_pad_truncate(input_tokens, input_lengths, fixed_len, pad_token)

    print("2. Running Custom CUDA Kernel Implementation...")
    custom_out = custom_cuda_ops.pad_truncate(input_tokens, input_lengths, fixed_len, pad_token)

    print("\n--- Comparison ---")
    
    # Compare element-wise
    are_equal = torch.equal(pytorch_out, custom_out)
    
    if are_equal:
        print("[SUCCESS] The custom CUDA kernel matches the normal PyTorch output perfectly!")
    else:
        print("[FAILED] The outputs do not match.")
        
    print("\nPyTorch Output Shape:", pytorch_out.shape)
    print("CUDA Kernel Output Shape:", custom_out.shape)

    print("\nInspecting Sentence 1 (Length 50):")
    print("PyTorch:", pytorch_out[0, 48:52]) # Should show 2 real tokens then 2 zeros
    print("CUDA:   ", custom_out[0, 48:52])

    print("\nInspecting Sentence 3 (Length 150 - Truncated to 128):")
    print("PyTorch End:", pytorch_out[2, 125:128])
    print("CUDA End:   ", custom_out[2, 125:128])

if __name__ == "__main__":
    main()
