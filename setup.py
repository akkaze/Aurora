from setuptools import setup, find_packages
from cffi import FFI
import os
import numpy

# ---------- CFFI 扩展 ----------
ffi = FFI()

# 声明所有需要导出的 C 函数（与您的 C 源文件一致）
ffi.cdef("""
    void im2col_c(const double *image,
                  int N, int C, int H, int W,
                  int filter_h, int filter_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w,
                  double *out);
    void col2im_c(const double *cols,
                  int N, int C, int H, int W,
                  int filter_h, int filter_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w,
                  double *out);
    void max_pool_forward_c(const double *data,
                            int N, int C, int H, int W,
                            int filter_h, int filter_w,
                            int stride_h, int stride_w,
                            double *out);
    void max_pool_backward_c(const double *dy, const double *x,
                             int N, int C, int H, int W,
                             int filter_h, int filter_w,
                             int stride_h, int stride_w,
                             double *grad);
""")

# 设置源文件：合并两个 C 文件（或者分别包含）
c_dir = os.path.join(os.path.dirname(__file__), 'aurora', 'nn', 'c')
ffi.set_source('aurora.nn._fast_ops',
    """
    #include "im2col.c"
    #include "fast_pooling.c"
    """,
    include_dirs=[c_dir],
    # 编译优化选项（根据需求调整）
    extra_compile_args=['-O3', '-march=native', '-fopenmp'],
    extra_link_args=['-fopenmp']
)

setup(
    name='aurora',
    version='0.01',
    description='Minimal Deep Learning library...',
    author='Upul Bandara',
    author_email='upulbandara@gmail.com',
    license='MIT',
    # 仅包含 CFFI 扩展（不再有 Cython 扩展）
    ext_modules=[ffi.distutils_extension()],
    packages=find_packages(exclude=['Aurora.tests']),
    install_requires=['cffi', 'numpy'],
    # 确保 C 源文件被打包（以便在源码分发时可用）
    package_data={'aurora.nn.c': ['*.c']},
    include_package_data=True,
)