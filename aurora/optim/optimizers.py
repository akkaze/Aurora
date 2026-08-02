import numpy as np
import aurora.autodiff as ad
import time


def build_sgd_updates(cost, params, lr=0.1, momentum=0.9):
    grads = ad.gradients(cost, params)
    velocities = [ad.Variable(name=f"{p.name}_vel") for p in params]
    lr_node = ad.Const(name="lr", value=np.array([lr], dtype=np.float32))
    momentum_node = ad.Const(
        name="momentum", value=np.array([momentum], dtype=np.float32)
    )

    new_params = []
    new_vels = []
    for p, g, v in zip(params, grads, velocities):
        # 显式广播到参数形状
        lr_b = ad.broadcast_to(lr_node, p)
        momentum_b = ad.broadcast_to(momentum_node, p)
        v_new = momentum_b * v - lr_b * g
        p_new = p + v_new
        new_params.append(p_new)
        new_vels.append(v_new)
    return new_params, new_vels, (velocities,)


def build_adam_updates(cost, params, lr=1e-3, beta1=0.9, beta2=0.995, eps=1e-8):
    # 1. 计算梯度
    grads = ad.gradients(cost, params)

    # 2. 状态变量
    t_node = ad.Variable(name="adam_t")  # 迭代步数（从 1 开始）
    beta1_t_node = ad.Variable(name="adam_beta1_t")  # beta1^t
    beta2_t_node = ad.Variable(name="adam_beta2_t")  # beta2^t
    momentums = [ad.Variable(name=f"{p.name}_mom") for p in params]
    velocities = [ad.Variable(name=f"{p.name}_vel") for p in params]

    # 3. 超参数常量（标量）
    lr_node = ad.Const(name="lr", value=np.array([lr], dtype=np.float32))
    eps_node = ad.Const(name="eps", value=np.array([eps], dtype=np.float32))
    beta1_node = ad.Const(name="beta1", value=np.array([beta1], dtype=np.float32))
    beta2_node = ad.Const(name="beta2", value=np.array([beta2], dtype=np.float32))

    # 4. 用于构造 (1 - ...) 的全 1 标量常量（后续广播）
    ones_scalar = ad.Const(name="ones", value=np.array([1.0], dtype=np.float32))

    new_params = []
    new_moms = []
    new_vels = []

    # 收集调试节点（若 debug=True）
    debug_nodes = {}

    for idx, (p, g, m, v) in enumerate(zip(params, grads, momentums, velocities)):
        # --- 广播超参数到当前参数形状 ---
        lr_b = ad.broadcast_to(lr_node, p)
        eps_b = ad.broadcast_to(eps_node, p)
        beta1_b = ad.broadcast_to(beta1_node, p)
        beta2_b = ad.broadcast_to(beta2_node, p)
        beta1_t_b = ad.broadcast_to(beta1_t_node, p)
        beta2_t_b = ad.broadcast_to(beta2_t_node, p)
        ones_b = ad.broadcast_to(ones_scalar, p)  # 全 1 张量节点

        # --- 一阶、二阶动量更新 ---
        # 注意：使用 ones_b - beta1_b 而不是 1 - beta1_b
        m_new = beta1_b * m + (ones_b - beta1_b) * g
        v_new = beta2_b * v + (ones_b - beta2_b) * (g * g)

        # --- 偏差校正 ---
        # 分母 (1 - beta^t)，使用 ones_b - beta_t_b
        m_hat = m_new / (ones_b - beta1_t_b)
        v_hat = v_new / (ones_b - beta2_t_b)

        # --- 参数更新 ---
        # 分母 sqrt(v_hat) + eps
        denom = ad.sqrt(v_hat) + eps_b
        p_new = p - lr_b * m_hat / denom

        # --- 保存更新节点 ---
        new_params.append(p_new)
        new_moms.append(m_new)
        new_vels.append(v_new)

    state_nodes = (t_node, beta1_t_node, beta2_t_node, momentums, velocities)

    return new_params, new_moms, new_vels, state_nodes


def build_optimizer_updates(optim_type, cost, params, **kwargs):
    if optim_type == "sgd":
        return build_sgd_updates(cost, params, **kwargs)
    elif optim_type == "adam":
        return build_adam_updates(cost, params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")
