#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h> /* 可选，若不需要 OpenMP 可去掉 */

#define BIG_NEGATIVE -1.0e15

/* ---------- 池化前向 ---------- */
void max_pool_forward_c(const double *data,
                        int N, int C, int H, int W,
                        int filter_h, int filter_w,
                        int stride_h, int stride_w,
                        double *out)
{
    int H_out = (H - filter_h) / stride_h + 1;
    int W_out = (W - filter_w) / stride_w + 1;
    double max_val;
#pragma omp parallel for collapse(2) private(max_val)
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int v_start = h * stride_h;
                    int h_end = v_start + filter_h;
                    int w_start = w * stride_w;
                    int w_end = w_start + filter_w;
                    max_val = BIG_NEGATIVE;
                    for (int ph = v_start; ph < h_end; ++ph)
                        for (int pw = w_start; pw < w_end; ++pw)
                        {
                            double val = data[(((n * C) + c) * H + ph) * W + pw];
                            if (val > max_val)
                                max_val = val;
                        }
                    out[(((n * C) + c) * H_out + h) * W_out + w] = max_val;
                }
            }
        }
    }
}

/* ---------- 池化反向 ---------- */
void max_pool_backward_c(const double *dy, const double *x,
                         int N, int C, int H, int W,
                         int filter_h, int filter_w,
                         int stride_h, int stride_w,
                         double *grad)
{
    int H_out = (H - filter_h) / stride_h + 1;
    int W_out = (W - filter_w) / stride_w + 1;
    memset(grad, 0, N * C * H * W * sizeof(double));

#pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int v_start = h * stride_h;
                    int h_end = v_start + filter_h;
                    int w_start = w * stride_w;
                    int w_end = w_start + filter_w;
                    double max_val = BIG_NEGATIVE;
                    int max_i = v_start, max_j = w_start;
                    for (int ph = v_start; ph < h_end; ++ph)
                        for (int pw = w_start; pw < w_end; ++pw)
                        {
                            double val = x[(((n * C) + c) * H + ph) * W + pw];
                            if (val > max_val)
                            {
                                max_val = val;
                                max_i = ph;
                                max_j = pw;
                            }
                        }
                    double dy_val = dy[(((n * C) + c) * H_out + h) * W_out + w];
                    grad[(((n * C) + c) * H + max_i) * W + max_j] += dy_val;
                }
            }
        }
    }
}