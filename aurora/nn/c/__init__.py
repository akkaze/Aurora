import numpy as np
from aurora.nn._fast_ops import lib, ffi

def _ensure_contiguous_float64(arr):
    return np.ascontiguousarray(arr, dtype=np.float64)

def im2col(image, filter_height=3, filter_width=3,
           padding_height=0, padding_width=0,
           stride_height=1, stride_width=1):
    image = _ensure_contiguous_float64(image)
    
    N, C, H, W = image.shape
    H_pad = H + 2 * padding_height
    W_pad = W + 2 * padding_width
    new_h = (H_pad - filter_height) // stride_height + 1
    new_w = (W_pad - filter_width) // stride_width + 1
    col_h = filter_height * filter_width * C
    col_w = N * new_h * new_w
    out = np.zeros((col_h, col_w), dtype=np.float64)
    lib.im2col_c(
        ffi.cast("const double*", image.ctypes.data),
        N, C, H, W,
        filter_height, filter_width,
        padding_height, padding_width,
        stride_height, stride_width,
        ffi.cast("double*", out.ctypes.data)
    )
    return out

def col2im(cols, batch_size, no_channels, image_height, image_width,
           filter_height=3, filter_width=3,
           padding_height=0, padding_width=0,
           stride_height=1, stride_width=1):
    cols = _ensure_contiguous_float64(cols)
    out = np.zeros((batch_size, no_channels, image_height, image_width),
                   dtype=np.float64)
    lib.col2im_c(
        ffi.cast("const double*", cols.ctypes.data),
        batch_size, no_channels, image_height, image_width,
        filter_height, filter_width,
        padding_height, padding_width,
        stride_height, stride_width,
        ffi.cast("double*", out.ctypes.data)
    )
    return out

def max_pool_forward(data, filter_height, filter_width,
                     stride_height, stride_width):
    data = _ensure_contiguous_float64(data)
    N, C, H, W = data.shape
    H_out = (H - filter_height) // stride_height + 1
    W_out = (W - filter_width) // stride_width + 1
    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    lib.max_pool_forward_c(
        ffi.cast("const double*", data.ctypes.data),
        N, C, H, W,
        filter_height, filter_width,
        stride_height, stride_width,
        ffi.cast("double*", out.ctypes.data)
    )
    return out

def max_pool_backward(dy, x, filter_height, filter_width,
                      stride_height, stride_width):
    dy = _ensure_contiguous_float64(dy)
    x = _ensure_contiguous_float64(x)
    N, C, H, W = x.shape
    grad = np.zeros_like(x)
    lib.max_pool_backward_c(
        ffi.cast("const double*", dy.ctypes.data),
        ffi.cast("const double*", x.ctypes.data),
        N, C, H, W,
        filter_height, filter_width,
        stride_height, stride_width,
        ffi.cast("double*", grad.ctypes.data)
    )
    return grad