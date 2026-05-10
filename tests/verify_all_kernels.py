import torch
import custom_cuda_ops
import math
import numpy as np

def test_kernel(name, func, *args, rtol=1e-3, atol=1e-5):
    print(f"Verifying {name}...", end="", flush=True)
    try:
        torch_out, custom_out = func(*args)
        torch.testing.assert_close(custom_out.float(), torch_out.float(), rtol=rtol, atol=atol)
        print(" PASS")
    except Exception as e:
        print(f" FAIL\n  Error: {e}")

# --- Test Functions ---

def verify_pad_truncate():
    batch, input_stride, fixed_len = 4, 160, 128
    tokens = torch.randint(0, 1000, (batch, input_stride), dtype=torch.int32).cuda()
    lengths = torch.tensor([10, 128, 150, 5], dtype=torch.int32).cuda()
    
    custom_out = custom_cuda_ops.pad_truncate(tokens, lengths, fixed_len, 0)
    
    expected = torch.zeros((batch, fixed_len), dtype=torch.int32).cuda()
    for i in range(batch):
        L = min(lengths[i].item(), fixed_len)
        expected[i, :L] = tokens[i, :L]
    return expected, custom_out

def verify_embedding_lookup():
    batch, seq_len, dim, vocab = 2, 128, 64, 1000
    tokens = torch.randint(0, vocab, (batch, seq_len), dtype=torch.int32).cuda()
    emb_weight = torch.randn(vocab, dim).cuda()
    
    custom_out = custom_cuda_ops.embedding_lookup(tokens.view(-1), emb_weight, 1)
    
    expected = torch.nn.functional.embedding(tokens, emb_weight).view(-1, dim)
    return expected, custom_out

def verify_bias_add():
    rows, cols = 128, 64
    x = torch.randn(rows, cols).cuda()
    bias = torch.randn(cols).cuda()
    
    custom_out = custom_cuda_ops.bias_add(x, bias)
    return x + bias, custom_out

def verify_leaky_relu():
    x = torch.randn(1024).cuda()
    alpha = 0.01
    custom_out = custom_cuda_ops.leaky_relu(x, alpha)
    return torch.nn.functional.leaky_relu(x, alpha), custom_out

def verify_gemm():
    M, K, N = 128, 64, 128
    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()
    custom_out = custom_cuda_ops.gemm_tiled(A, B)
    return torch.matmul(A, B), custom_out

def verify_softmax():
    batch, classes = 128, 10
    x = torch.randn(batch, classes).cuda()
    
    row_max = custom_cuda_ops.softmax_row_max(x)
    row_sum = custom_cuda_ops.softmax_row_sum(x, row_max)
    custom_out = custom_cuda_ops.softmax_normalize(x, row_max, row_sum)
    
    return torch.nn.functional.softmax(x, dim=-1), custom_out

def verify_argmax():
    batch, classes = 128, 10
    x = torch.randn(batch, classes).cuda()
    custom_out = custom_cuda_ops.argmax(x)
    return torch.argmax(x, dim=-1).to(torch.int32), custom_out

if __name__ == "__main__":
    print("=== Unified Kernel Verification ===")
    test_kernel("K1: Pad/Truncate", verify_pad_truncate)
    test_kernel("K2: Embedding", verify_embedding_lookup)
    test_kernel("K5: Bias Add", verify_bias_add)
    test_kernel("K6: Leaky ReLU", verify_leaky_relu)
    test_kernel("K10: GEMM Tiled", verify_gemm)
    test_kernel("K12-14: Softmax", verify_softmax)
    test_kernel("K15: Argmax", verify_argmax)
    print("===================================")
