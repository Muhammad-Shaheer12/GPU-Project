import torch
import custom_cuda_ops
import torch.utils.benchmark as benchmark

def benchmark_kernel(name, func, *args):
    print(f"Benchmarking {name}...")
    t = benchmark.Timer(
        stmt='func(*args)',
        globals={'func': func, 'args': args},
        num_threads=1
    )
    m = t.blocked_autorange(min_run_time=1)
    print(f"  Result: {m.median * 1e6:>7.2f} us")
    return m.median

# --- Wrappers for Benchmarking ---

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

def bench_gemm():
    M, K, N = 2048, 64, 128
    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()
    return custom_cuda_ops.gemm_tiled, A, B

def bench_softmax_full():
    batch, classes = 2048, 10
    x = torch.randn(batch, classes).cuda()
    
    def full_softmax(x):
        row_max = custom_cuda_ops.softmax_row_max(x)
        row_sum = custom_cuda_ops.softmax_row_sum(x, row_max)
        return custom_cuda_ops.softmax_normalize(x, row_max, row_sum)
    
    return full_softmax, x

if __name__ == "__main__":
    print("=== Unified Kernel Benchmarking (Median Execution Time) ===")
    benchmark_kernel("K1: Pad/Truncate", *bench_pad_truncate())
    benchmark_kernel("K2: Embedding", *bench_embedding_lookup())
    benchmark_kernel("K10: GEMM Tiled", *bench_gemm())
    benchmark_kernel("K12-14: Full Softmax", *bench_softmax_full())
    print("==========================================================")
