from setuptools import setup
import torch.utils.cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Bypass the strict CUDA version check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

kernals_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Kernals'))

setup(
    name='custom_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_cuda_ops',
            sources=[
                'custom_ops.cpp',
                'custom_kernels_wrapper.cu',
                os.path.join(kernals_dir, 'kernel1_pad_truncate.cu'),
                os.path.join(kernals_dir, 'kernel2_embedding_lookup.cu'),
                os.path.join(kernals_dir, 'kernel3_positional_encoding.cu'),
                os.path.join(kernals_dir, 'kernel4_weighted_mean_pooling.cu'),
                os.path.join(kernals_dir, 'kernel5_bias_add.cu'),
                os.path.join(kernals_dir, 'kernel6_leaky_relu.cu'),
                os.path.join(kernals_dir, 'kernel7_batchnorm_mean.cu'),
                os.path.join(kernals_dir, 'kernel8_batchnorm_var.cu'),
                os.path.join(kernals_dir, 'kernel9_batchnorm_apply.cu'),
                os.path.join(kernals_dir, 'kernel10_gemm_tiled.cu'),
                os.path.join(kernals_dir, 'kernel11_logit_projection.cu'),
                os.path.join(kernals_dir, 'kernel12_softmax_row_max.cu'),
                os.path.join(kernals_dir, 'kernel13_softmax_row_sum.cu'),
                os.path.join(kernals_dir, 'kernel14_softmax_normalize.cu'),
                os.path.join(kernals_dir, 'kernel15_argmax.cu'),
            ],
            include_dirs=[kernals_dir],
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3', '-DPIPELINE_BUILD']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
