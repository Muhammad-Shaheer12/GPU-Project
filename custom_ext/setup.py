from setuptools import setup
import torch.utils.cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Bypass the strict CUDA version check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None


setup(
    name='custom_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_cuda_ops',
            sources=[
                'custom_ops.cpp',
                'custom_kernels_wrapper.cu',
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kernals_stripped', 'kernel1_pad_truncate.cu')),
            ],
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
