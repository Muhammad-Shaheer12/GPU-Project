import torch
import os
import sys

# Setup paths for Windows DLL loading and local extension
ext_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_ext'))
sys.path.append(ext_path)

if sys.platform == 'win32':
    # Add torch lib for DLLs like torch_python.dll and cublas
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
    
    # Add CUDA bin for cublas and other runtimes
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path:
        # On Windows, DLLs are often in bin/x64 for CUDA 12+
        cuda_bin = os.path.join(cuda_path, 'bin', 'x64')
        if not os.path.exists(cuda_bin):
            cuda_bin = os.path.join(cuda_path, 'bin')
        
        if os.path.exists(cuda_bin):
            os.add_dll_directory(cuda_bin)

    # Add extension path for the .pyd itself
    os.add_dll_directory(ext_path)

import custom_cuda_ops
import torch.utils.benchmark as benchmark
import math
import numpy as np

def run_test(name, verify_func=None, bench_args_func=None, rtol=1e-3, atol=1e-5):
    print(f"--- {name} ---")
    verified = True
    if verify_func:
        print(f"  Verifying...", end="", flush=True)
        try:
            torch_out, custom_out = verify_func()
            torch.testing.assert_close(custom_out.float(), torch_out.float(), rtol=rtol, atol=atol)
            print(" PASS")
        except Exception as e:
            print(f" FAIL\n    Error: {e}")
            verified = False
            print("  Skipping benchmark due to verification failure.")
    else:
        print("  Verifying... N/A (No verification test defined)")
    
    if verified and bench_args_func:
        print(f"  Benchmarking...", end="", flush=True)
        try:
            func_and_args = bench_args_func()
            func = func_and_args[0]
            args = func_and_args[1:]
            
            t = benchmark.Timer(
                stmt='func(*args)',
                globals={'func': func, 'args': args},
                num_threads=1
            )
            m = t.blocked_autorange(min_run_time=1)
            print(f" {m.median * 1e6:>7.2f} us")
        except Exception as e:
            print(f" FAIL\n    Error: {e}")
    elif not bench_args_func:
        print("  Benchmarking... N/A")

# Verification Functions
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

def verify_positional_encoding():
    total_tokens, dim = 2048, 64
    x = torch.randn(total_tokens, dim).cuda()
    custom_out = custom_cuda_ops.positional_encoding(x)
    
    positions = torch.arange(total_tokens, dtype=torch.float32).unsqueeze(1).cuda()
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)).cuda()
    pe = torch.zeros(total_tokens, dim).cuda()
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    
    expected = x + pe
    return expected, custom_out

def verify_weighted_mean_pooling():
    batch, seq_len, dim = 128, 64, 64
    x = torch.randn(batch, seq_len, dim).cuda()
    w = torch.rand(batch, seq_len).cuda()
    custom_out = custom_cuda_ops.weighted_mean_pooling(x, w)
    
    expected = (x * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1).unsqueeze(-1)
    return expected, custom_out

def verify_fused_bias_relu():
    rows, cols = 128, 64
    x = torch.randn(rows, cols).cuda()
    b = torch.randn(cols).cuda()
    alpha = 0.01
    custom_out = custom_cuda_ops.fused_bias_leaky_relu(x, b, alpha)
    expected = torch.nn.functional.leaky_relu(x + b, alpha)
    return expected, custom_out

def verify_batchnorm_full():
    batch, features = 128, 64
    x = torch.randn(batch, features).cuda()
    gamma = torch.randn(features).cuda()
    beta = torch.randn(features).cuda()
    
    mean = custom_cuda_ops.batchnorm_mean(x)
    var = custom_cuda_ops.batchnorm_var(x, mean)
    custom_out = custom_cuda_ops.batchnorm_apply(x, mean, var, gamma, beta, 1e-5)
    
    expected = torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=gamma, bias=beta, training=True, momentum=0.0, eps=1e-5)
    return expected, custom_out

def verify_logit_projection():
    batch, hidden, classes = 128, 64, 5
    x = torch.randn(batch, hidden).cuda()
    w = torch.randn(hidden, classes).cuda()
    custom_out = custom_cuda_ops.logit_projection(x, w)
    expected = torch.matmul(x, w)
    return expected, custom_out

def verify_softmax_fused():
    batch, classes = 128, 5
    x = torch.randn(batch, classes).cuda()
    custom_out = custom_cuda_ops.fused_softmax(x)
    expected = torch.nn.functional.softmax(x, dim=-1)
    return expected, custom_out

# Benchmarking Wrappers
def bench_pad_truncate():
    batch, input_stride, fixed_len = 2048, 160, 128
    tokens = torch.randint(0, 1000, (batch, input_stride), dtype=torch.int32).cuda()
    lengths = torch.randint(1, 160, (batch,), dtype=torch.int32).cuda()
    return custom_cuda_ops.pad_truncate, tokens, lengths, fixed_len, 0

def bench_embedding_lookup():
    total_tokens, dim, vocab = 2048 * 128, 64, 103569
    tokens = torch.randint(0, vocab, (total_tokens,), dtype=torch.int32).cuda()
    emb_weight = torch.randn(vocab, dim).cuda()
    return custom_cuda_ops.embedding_lookup, tokens, emb_weight, 1

def bench_positional_encoding():
    total_tokens, dim = 2048 * 128, 64
    x = torch.randn(total_tokens, dim).cuda()
    return custom_cuda_ops.positional_encoding, x

def bench_weighted_mean_pooling():
    batch, seq_len, dim = 2048, 128, 64
    x = torch.randn(batch, seq_len, dim).cuda()
    w = torch.randn(batch, seq_len).cuda()
    return custom_cuda_ops.weighted_mean_pooling, x, w

def bench_bias_add():
    rows, cols = 2048 * 128, 64
    x = torch.randn(rows, cols).cuda()
    b = torch.randn(cols).cuda()
    return custom_cuda_ops.bias_add, x, b

def bench_leaky_relu():
    x = torch.randn(2048 * 128, 64).cuda()
    return custom_cuda_ops.leaky_relu, x, 0.01

def bench_fused_bias_relu():
    rows, cols = 2048 * 128, 64
    x = torch.randn(rows, cols).cuda()
    b = torch.randn(cols).cuda()
    return custom_cuda_ops.fused_bias_leaky_relu, x, b, 0.01

def bench_batchnorm_full():
    batch, features = 2048, 128
    x = torch.randn(batch, features).cuda()
    gamma = torch.randn(features).cuda()
    beta = torch.randn(features).cuda()
    def full_bn(x, gamma, beta):
        mean = custom_cuda_ops.batchnorm_mean(x)
        var = custom_cuda_ops.batchnorm_var(x, mean)
        return custom_cuda_ops.batchnorm_apply(x, mean, var, gamma, beta, 1e-5)
    return full_bn, x, gamma, beta

def bench_gemm():
    M, K, N = 2048, 64, 128
    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()
    return custom_cuda_ops.gemm_tiled, A, B

def bench_logit_projection():
    batch, hidden, classes = 2048, 128, 5
    x = torch.randn(batch, hidden).cuda()
    w = torch.randn(hidden, classes).cuda()
    return custom_cuda_ops.logit_projection, x, w

def bench_softmax_unfused():
    batch, classes = 2048, 5
    x = torch.randn(batch, classes).cuda()
    def full_softmax(x):
        row_max = custom_cuda_ops.softmax_row_max(x)
        row_sum = custom_cuda_ops.softmax_row_sum(x, row_max)
        return custom_cuda_ops.softmax_normalize(x, row_max, row_sum)
    return full_softmax, x

def bench_softmax_fused():
    batch, classes = 2048, 5
    x = torch.randn(batch, classes).cuda()
    return custom_cuda_ops.fused_softmax, x

def bench_argmax():
    batch, classes = 2048, 5
    x = torch.randn(batch, classes).cuda()
    return custom_cuda_ops.argmax, x

if __name__ == "__main__":
    print("==========================================")
    print("=== Unified Kernel Testing & Profiling ===")
    print("==========================================")
    run_test("K1: Pad/Truncate", verify_pad_truncate, bench_pad_truncate)
    run_test("K2: Embedding", verify_embedding_lookup, bench_embedding_lookup)
    run_test("K3: Positional Encoding", verify_positional_encoding, bench_positional_encoding, atol=2e-4)
    run_test("K4: Weighted Pooling", verify_weighted_mean_pooling, bench_weighted_mean_pooling)
    run_test("K5: Bias Add", verify_bias_add, bench_bias_add)
    run_test("K6: Leaky ReLU", verify_leaky_relu, bench_leaky_relu)
    run_test("K16: Fused Bias+ReLU", verify_fused_bias_relu, bench_fused_bias_relu)
    run_test("K7-9: Full BatchNorm", verify_batchnorm_full, bench_batchnorm_full)
    run_test("K10: GEMM Tiled", verify_gemm, bench_gemm)
    run_test("K11: Logit Projection", verify_logit_projection, bench_logit_projection)
    run_test("K12-14: Unfused Softmax", verify_softmax, bench_softmax_unfused)
    run_test("K17: Fused Softmax", verify_softmax_fused, bench_softmax_fused)
    run_test("K15: Argmax", verify_argmax, bench_argmax)
    print("==========================================")
