import cupy as cp

_im2col_kernel_code = r'''
extern "C" __global__
void im2col_kernel(
    const double* __restrict__ x_padded,   // (N, C, H_pad, W_pad)
    double* __restrict__ out,              // (K, total) 列主序
    int N, int C, int H_pad, int W_pad,
    int filter_h, int filter_w,
    int stride_h, int stride_w,
    int h_new, int w_new
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * h_new * w_new;
    if (idx >= total) return;

    // 按照空间优先顺序解码：i -> j -> m
    int i = idx / (N * w_new);
    int rem = idx - i * N * w_new;
    int j = rem / N;
    int m = rem % N;

    int start_h = i * stride_h;
    int start_w = j * stride_w;

    int K = C * filter_h * filter_w;
    int base_in = m * C * H_pad * W_pad;

    int k = 0;
    for (int c = 0; c < C; ++c) {
        int in_chan_base = base_in + c * H_pad * W_pad;
        for (int ph = 0; ph < filter_h; ++ph) {
            int row = start_h + ph;
            for (int pw = 0; pw < filter_w; ++pw) {
                int col = start_w + pw;
                int in_idx = in_chan_base + row * W_pad + col;
                out[k * total + idx] = x_padded[in_idx];
                ++k;
            }
        }
    }
}
'''

_im2col_kernel = cp.RawKernel(_im2col_kernel_code, 'im2col_kernel')

def im2col_gpu(image, filter_height, filter_width,
               padding_height, padding_width,
               stride_height, stride_width):
    if image.dtype != cp.float64:
        image = image.astype(cp.float64)
    if image.ndim != 4:
        raise ValueError("Expected 4D input")
    N, C, H, W = image.shape

    H_pad = H + 2 * padding_height
    W_pad = W + 2 * padding_width
    h_new = (H_pad - filter_height) // stride_height + 1
    w_new = (W_pad - filter_width) // stride_width + 1
    if h_new <= 0 or w_new <= 0:
        raise ValueError("Invalid dimensions")

    x_padded = cp.pad(
        image,
        ((0,0), (0,0), (padding_height, padding_height), (padding_width, padding_width)),
        mode='constant'
    )

    K = C * filter_height * filter_width
    total = N * h_new * w_new
    out = cp.zeros((K, total), dtype=cp.float64)

    threads = 256
    blocks = (total + threads - 1) // threads
    if total > 0:
        _im2col_kernel(
            (blocks,), (threads,),
            (x_padded, out, N, C, H_pad, W_pad,
             filter_height, filter_width,
             stride_height, stride_width,
             h_new, w_new)
        )
    return out

# ---------- col2im kernel ----------
_col2im_kernel_code = r'''
extern "C" __global__
void col2im_kernel(
    const double* __restrict__ cols,      // shape: (K, N)
    double* __restrict__ x_padded,        // shape: (M, C, H_pad, W_pad)
    int M, int C, int H_pad, int W_pad,
    int filter_h, int filter_w,
    int stride_h, int stride_w,
    int h_new, int w_new
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * h_new * w_new;
    if (idx >= total) return;

    int m = idx / (h_new * w_new);
    int remainder = idx % (h_new * w_new);
    int i = remainder / w_new;
    int j = remainder % w_new;

    int start_i = i * stride_h;
    int start_j = j * stride_w;
    int K = C * filter_h * filter_w;

    // 对于每个输出位置，累加对应列到 x_padded 的窗口区域
    int base_col = idx;  // 列索引
    int base_in = m * C * H_pad * W_pad;

    for (int c = 0; c < C; ++c) {
        int in_chan_base = base_in + c * H_pad * W_pad;
        for (int ph = 0; ph < filter_h; ++ph) {
            int row = start_i + ph;
            for (int pw = 0; pw < filter_w; ++pw) {
                int col = start_j + pw;
                int in_idx = in_chan_base + row * W_pad + col;
                int out_idx = ( (c * filter_h + ph) * filter_w + pw ) * total + base_col;
                atomicAdd(&x_padded[in_idx], cols[out_idx]);
            }
        }
    }
}
'''

_col2im_kernel = cp.RawKernel(_col2im_kernel_code, 'col2im_kernel')

def col2im_gpu(cols, batch_size, no_channels, image_height, image_width,
               filter_height, filter_width,
               padding_height, padding_width,
               stride_height, stride_width):
    """
    GPU 版本的 col2im。
    输入: cols (cp.ndarray, shape=(K, M * h_new * w_new))
    输出: cp.ndarray, shape=(M, C, H, W)
    """
    if cols.dtype != cp.float64:
        cols = cols.astype(cp.float64)
    M = batch_size
    C = no_channels
    H = image_height
    W = image_width

    H_pad = H + 2 * padding_height
    W_pad = W + 2 * padding_width
    x_padded = cp.zeros((M, C, H_pad, W_pad), dtype=cp.float64)

    h_new = (H - filter_height + 2 * padding_height) // stride_height + 1
    w_new = (W - filter_width + 2 * padding_width) // stride_width + 1
    total = M * h_new * w_new

    threads = 256
    blocks = (total + threads - 1) // threads
    _col2im_kernel(
        (blocks,), (threads,),
        (cols, x_padded, M, C, H_pad, W_pad,
         filter_height, filter_width,
         stride_height, stride_width,
         h_new, w_new)
    )

    # 裁剪 padding 区域
    if padding_height or padding_width:
        return x_padded[:, :, padding_height:-padding_height, padding_width:-padding_width]
    else:
        return x_padded