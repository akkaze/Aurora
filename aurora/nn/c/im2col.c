#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h> /* 可选，若不需要 OpenMP 可去掉 */

#define BIG_NEGATIVE -1.0e15

/* ---------- im2col 核心 ---------- */
static void im2col_inner_c(const double *x_padded, /* 形状 (N, C, H_pad, W_pad) */
                           int N, int C, int H_pad, int W_pad,
                           double *out, /* 形状 (col_h, col_w) */
                           int filter_h, int filter_w,
                           int stride_h, int stride_w,
                           int h_new, int w_new)
{
    int col_h = filter_h * filter_w * C;
    int col_w = N * h_new * w_new;
    int i, j, m, c, p_h, p_w;
    int k, idx = 0;

    for (i = 0; i < h_new; ++i)
    {
        for (j = 0; j < w_new; ++j)
        {
            int start_h = i * stride_h;
            int end_h = start_h + filter_h;
            int start_w = j * stride_w;
            int end_w = start_w + filter_w;
            for (m = 0; m < N; ++m)
            {
                k = 0;
                for (c = 0; c < C; ++c)
                {
                    for (p_h = start_h; p_h < end_h; ++p_h)
                    {
                        for (p_w = start_w; p_w < end_w; ++p_w)
                        {
                            double val = x_padded[(((m * C) + c) * H_pad + p_h) * W_pad + p_w];
                            out[k * col_w + idx] = val;
                            ++k;
                        }
                    }
                }
                ++idx;
            }
        }
    }
}

/* ---------- col2im 核心 ---------- */
static void col2im_inner_c(const double *cols, /* 形状 (col_h, col_w) */
                           double *x_padded,   /* 形状 (N, C, H_pad, W_pad) */
                           int N, int C, int H_pad, int W_pad,
                           int filter_h, int filter_w,
                           int stride_h, int stride_w,
                           int h_new, int w_new)
{
    int col_h = filter_h * filter_w * C;
    int col_w = N * h_new * w_new;
    int i, j, m, c, p_h, p_w;
    int k, idx = 0;

    // 初始化 x_padded 为 0（调用者已清零）
    for (i = 0; i < h_new; ++i)
    {
        for (j = 0; j < w_new; ++j)
        {
            int start_h = i * stride_h;
            int end_h = start_h + filter_h;
            int start_w = j * stride_w;
            int end_w = start_w + filter_w;
            for (m = 0; m < N; ++m)
            {
                k = 0;
                for (c = 0; c < C; ++c)
                {
                    for (p_h = start_h; p_h < end_h; ++p_h)
                    {
                        for (p_w = start_w; p_w < end_w; ++p_w)
                        {
                            double val = cols[k * col_w + idx];
                            x_padded[(((m * C) + c) * H_pad + p_h) * W_pad + p_w] += val;
                            ++k;
                        }
                    }
                }
                ++idx;
            }
        }
    }
}

/* ---------- 对外接口：im2col ---------- */
void im2col_c(const double *image, /* (N, C, H, W) */
              int N, int C, int H, int W,
              int filter_h, int filter_w,
              int pad_h, int pad_w,
              int stride_h, int stride_w,
              double *out)
{ /* 输出矩阵，已经分配好 */
    int H_pad = H + 2 * pad_h;
    int W_pad = W + 2 * pad_w;
    int h_new = (H_pad - filter_h) / stride_h + 1;
    int w_new = (W_pad - filter_w) / stride_w + 1;

    // 创建填充后的临时数组
    double *x_padded = (double *)calloc(N * C * H_pad * W_pad, sizeof(double));
    if (!x_padded)
        return;

    // 复制并填充（边缘零填充）
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                {
                    int src_idx = ((n * C + c) * H + h) * W + w;
                    int dst_idx = ((n * C + c) * H_pad + (h + pad_h)) * W_pad + (w + pad_w);
                    x_padded[dst_idx] = image[src_idx];
                }

    im2col_inner_c(x_padded, N, C, H_pad, W_pad,
                   out, filter_h, filter_w, stride_h, stride_w,
                   h_new, w_new);
    free(x_padded);
}

/* ---------- 对外接口：col2im ---------- */
void col2im_c(const double *cols, /* (col_h, col_w) */
              int N, int C, int H, int W,
              int filter_h, int filter_w,
              int pad_h, int pad_w,
              int stride_h, int stride_w,
              double *out)
{ /* 输出图像 (N, C, H, W) */
    int H_pad = H + 2 * pad_h;
    int W_pad = W + 2 * pad_w;
    int h_new = (H_pad - filter_h) / stride_h + 1;
    int w_new = (W_pad - filter_w) / stride_w + 1;

    double *x_padded = (double *)calloc(N * C * H_pad * W_pad, sizeof(double));
    if (!x_padded)
        return;

    col2im_inner_c(cols, x_padded, N, C, H_pad, W_pad,
                   filter_h, filter_w, stride_h, stride_w,
                   h_new, w_new);

    // 裁剪填充部分
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                {
                    int src_idx = ((n * C + c) * H_pad + (h + pad_h)) * W_pad + (w + pad_w);
                    int dst_idx = ((n * C + c) * H + h) * W + w;
                    out[dst_idx] = x_padded[src_idx];
                }
    free(x_padded);
}
