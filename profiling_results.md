# Kernel 1 (`pad_truncate`) Profiling Results

Here are the official metrics pulled from the GPU using the NVIDIA and PyTorch profiling tools for the custom PyTorch C++ Extension.

### Hardware & Profiling Metrics

| Metric | Tool Used | Result | What it means for this kernel |
| :--- | :--- | :--- | :--- |
| **Execution Time** | `torch.utils.benchmark` | **30.24 µs** *(vs 203ms in PyTorch)* | The custom C++ CUDA kernel drastically outperforms the Python PyTorch equivalent, running ~6,700x faster! |
| **Execution Time (Raw GPU)** | `ncu` | **12.19 µs** | The pure hardware execution time of the kernel itself (without the PyTorch wrapper overhead) is incredibly fast. |
| **Memory Bandwidth** | `ncu` | **55.50 GB/s** (19.39% of Peak) | Padding is a memory-bound operation, but because the arrays are so small, it finishes before saturating the total peak bandwidth of your GPU. |
| **Occupancy** | `ncu` | **74.29%** Achieved (100% Theoretical) | Excellent utilization! You are keeping the Streaming Multiprocessors (SMs) busy with a solid amount of active warps. |
| **Arithmetic Intensity** | `ncu` | **Compute**: 23.40%<br>**Memory**: 19.39% | As expected for a padding kernel, compute throughput is low because there is almost no math (just copying bytes and inserting zeros). |

### Where to find the test scripts
All testing scripts are located in the `GPUProject/tests/` folder:
- **`profile_ncu.py`**: Run `ncu --set full python tests/profile_ncu.py` to trace the deep-dive hardware metrics.
- **`profile_kernel1.py`**: Run `python tests/profile_kernel1.py` to run the `torch.utils.benchmark` speed comparison.
- **`compare_kernel1.py`**: Run `python tests/compare_kernel1.py` to verify the mathematical outputs are identical to native PyTorch.
