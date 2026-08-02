import cupy as cp

# ---------- forward kernel ----------
_max_pool_forward_kernel_code = r'''
extern "C" __global__
void max_pool_forward_kernel(
    const double* __restrict__ input,  // (N, C, H, W)
    double* __restrict__ output,        // (N, C, H_out, W_out)
    int N, int C, int H, int W,
    int filter_h, int filter_w,
    int stride_h, int stride_w,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // 解码索引
    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;

    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;

    double max_val = -1e15f;
    int base = (n * C + c) * H * W;
    for (int ph = 0; ph < filter_h; ++ph) {
        for (int pw = 0; pw < filter_w; ++pw) {
            int h_in = h_start + ph;
            int w_in = w_start + pw;
            double val = input[base + h_in * W + w_in];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}
'''

_max_pool_forward_kernel = cp.RawKernel(_max_pool_forward_kernel_code, 'max_pool_forward_kernel')

def max_pool_forward_gpu(input_data, filter_height, filter_width, stride_height, stride_width):
    """
    GPU 版本的 max pooling forward。
    输入: input_data (cp.ndarray, shape=(N, C, H, W)), dtype=float64
    输出: cp.ndarray, shape=(N, C, H_out, W_out)
    """
    if input_data.dtype != cp.float64:
        input_data = input_data.astype(cp.float64)
    N, C, H, W = input_data.shape
    H_out = (H - filter_height) // stride_height + 1
    W_out = (W - filter_width) // stride_width + 1
    output = cp.empty((N, C, H_out, W_out), dtype=cp.float64)

    total = N * C * H_out * W_out
    threads = 256
    blocks = (total + threads - 1) // threads

    _max_pool_forward_kernel(
        (blocks,), (threads,),
        (input_data, output, N, C, H, W,
         filter_height, filter_width,
         stride_height, stride_width,
         H_out, W_out)
    )
    return output


# ---------- backward kernel ----------
_max_pool_backward_kernel_code = r'''
extern "C" __global__
void max_pool_backward_kernel(
    const double* __restrict__ output_grad, // (N, C, H_out, W_out)
    const double* __restrict__ input_data,  // (N, C, H, W)
    double* __restrict__ grad_input,        // (N, C, H, W)
    int N, int C, int H, int W,
    int filter_h, int filter_w,
    int stride_h, int stride_w,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;

    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;
    int base = (n * C + c) * H * W;

    // 寻找最大值位置
    double max_val = -1e15f;
    int max_h = h_start, max_w = w_start;
    for (int ph = 0; ph < filter_h; ++ph) {
        for (int pw = 0; pw < filter_w; ++pw) {
            int h_in = h_start + ph;
            int w_in = w_start + pw;
            double val = input_data[base + h_in * W + w_in];
            if (val > max_val) {
                max_val = val;
                max_h = h_in;
                max_w = w_in;
            }
        }
    }

    // 累加梯度到对应位置
    int grad_idx = base + max_h * W + max_w;
    atomicAdd(&grad_input[grad_idx], output_grad[idx]);
}
'''

_max_pool_backward_kernel = cp.RawKernel(_max_pool_backward_kernel_code, 'max_pool_backward_kernel')

def max_pool_backward_gpu(output_grad, input_data, filter_height, filter_width,
                          stride_height, stride_width):
    """
    GPU 版本的 max pooling backward。
    输入: output_grad (cp.ndarray, (N,C,H_out,W_out)), input_data (cp.ndarray, (N,C,H,W))
    输出: grad_input (cp.ndarray, (N,C,H,W))
    """
    if output_grad.dtype != cp.float64:
        output_grad = output_grad.astype(cp.float64)
    if input_data.dtype != cp.float64:
        input_data = input_data.astype(cp.float64)

    N, C, H_out, W_out = output_grad.shape
    H, W = input_data.shape[2], input_data.shape[3]
    grad_input = cp.zeros_like(input_data, dtype=cp.float64)

    total = N * C * H_out * W_out
    threads = 256
    blocks = (total + threads - 1) // threads

    _max_pool_backward_kernel(
        (blocks,), (threads,),
        (output_grad, input_data, grad_input, N, C, H, W,
         filter_height, filter_width,
         stride_height, stride_width,
         H_out, W_out)
    )
    return grad_input