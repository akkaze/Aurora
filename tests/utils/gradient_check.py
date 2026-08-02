import cunumpy as xp
import numpy as np

def gradient_check_numpy_expr(func, x, output_gradient, h=1e-5):
    # 使用 cunumpy 提供的转换函数
    x_np = xp.to_numpy(x)
    out_grad_np = xp.to_numpy(output_gradient).astype(np.float64)
    grad_np = np.zeros_like(x_np, dtype=np.float64)
    it = np.nditer(x_np, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = x_np[idx]
        x_np[idx] = old + h
        x_pert = xp.to_cunumpy(x_np)  # 转换为当前后端数组
        pos = func(x_pert)
        pos_np = xp.to_numpy(pos).astype(np.float64)
        x_np[idx] = old - h
        x_pert = xp.to_cunumpy(x_np)
        neg = func(x_pert)
        neg_np = xp.to_numpy(neg).astype(np.float64)
        x_np[idx] = old
        grad_np[idx] = np.sum((pos_np - neg_np) * out_grad_np) / (2 * h)
        it.iternext()
    return xp.to_cunumpy(grad_np)