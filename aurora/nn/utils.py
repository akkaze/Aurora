import cunumpy as xp
import aurora.nn.c as c_fast_ops
import aurora.nn.cuda as cuda_fast_ops


def softmax_func(x):
    """
    Numerically stable softmax function. For more details
    about numerically calculations please refer:
    http://www.deeplearningbook.org/slides/04_numerical.pdf
    :param x:
    :return:
    """
    stable_values = x - xp.max(x, axis=1, keepdims=True)
    return xp.exp(stable_values) / xp.sum(xp.exp(stable_values), axis=1, keepdims=True)


def log_sum_exp(x):
    """
    log_sum_exp is a very useful function in machine learning.
    It can be seen in many places including cross-entropy error.
    However, the naive implementation is numerically unstable.
    Therefore, we use the following implementation. For more details
    please refer: http://www.deeplearningbook.org/slides/04_numerical.pdf
    :param x:
    :return:
    """
    mx = xp.max(x, axis=1, keepdims=True)
    safe = x - mx
    return mx + xp.log(xp.sum(xp.exp(safe), axis=1, keepdims=True))


# Following two methods were used in the initial version of the convolution operations.
# Later we introduced fast Cython versions of `im2col` and `col2im` implementations.
# Hence, these two methods are obsolete.
def im2col(
    image,
    filter_height=2,
    filter_width=2,
    padding_height=0,
    padding_width=0,
    stride_height=1,
    stride_width=1,
):
    if xp.is_cpu(image):
        return c_fast_ops.im2col(
            image,
            filter_height,
            filter_width,
            padding_height,
            padding_width,
            stride_height,
            stride_width,
        )
    else:
        return cuda_fast_ops.im2col_gpu(
            image,
            filter_height,
            filter_width,
            padding_height,
            padding_width,
            stride_height,
            stride_width,
        )


def col2im(
    cols,
    N, C, H, W,
    filter_height=2,
    filter_width=2,
    padding_height=0,
    padding_width=0,
    stride_height=1,
    stride_width=1,
):
    filter_height = filter_height
    filter_width = filter_width
    padding_height = padding_height
    padding_width = padding_width
    stride_height = stride_height
    stride_width = stride_width
    if xp.is_cpu(cols):
        return c_fast_ops.col2im(
            cols,
            N,
            C,
            H,
            W,
            filter_height,
            filter_width,
            padding_height,
            padding_width,
            stride_height,
            stride_width,
        )
    else:
        return cuda_fast_ops.col2im_gpu(
            cols,
            N,
            C,
            H,
            W,
            filter_height,
            filter_width,
            padding_height,
            padding_width,
            stride_height,
            stride_width,
        )


def max_pool_forward(x, filter_height, filter_width, stride_height=1, stride_width=1):
    if xp.is_cpu(x):
        return c_fast_ops.max_pool_forward(
            x, filter_height, filter_width, stride_height, stride_width
        )
    else:
        return cuda_fast_ops.max_pool_forward_gpu(
            x, filter_height, filter_width, stride_height, stride_width
        )


def max_pool_backward(
    dy, x, filter_height, filter_width, stride_height=1, stride_width=1
):
    if xp.is_cpu(x):
        return c_fast_ops.max_pool_backward(
            dy, x, filter_height, filter_width, stride_height, stride_width
        )
    else:
        return cuda_fast_ops.max_pool_backward_gpu(
            dy, x, filter_height, filter_width, stride_height, stride_width
        )
