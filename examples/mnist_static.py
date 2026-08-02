import cunumpy as xp
import aurora as au
import aurora.autodiff as ad
import timeit
import argparse
from aurora.autodiff.utils import to_d2lang
from aurora.optim import build_optimizer_updates


def measure_accuracy(activation, data, use_gpu=False):
    X_val, y_val = data
    X_val, y_val = xp.asarray(X_val), xp.asarray(y_val)
    executor = ad.Executor([activation], use_gpu=use_gpu)
    (prob_val,) = executor.run(feed_shapes={X: X_val})
    correct = xp.sum(xp.equal(y_val, xp.argmax(prob_val, axis=1)))
    return (correct / y_val.shape[0]) * 100.0


def build_graph(X, y, input_size, hid_1_size, hid_2_size, output_size):
    rand = xp.random.RandomState(seed=1024)
    W1 = ad.Parameter(
        name="W1", init=rand.normal(scale=0.1, size=(input_size, hid_1_size))
    )
    b1 = ad.Parameter(name="b1", init=rand.normal(scale=0.1, size=(hid_1_size)))
    W2 = ad.Parameter(
        name="W2", init=rand.normal(scale=0.1, size=(hid_1_size, hid_2_size))
    )
    b2 = ad.Parameter(name="b2", init=rand.normal(scale=0.1, size=(hid_2_size)))
    W3 = ad.Parameter(
        name="W3", init=rand.normal(scale=0.1, size=(hid_2_size, output_size))
    )
    b3 = ad.Parameter(name="b3", init=rand.normal(scale=0.1, size=(output_size)))

    z1 = ad.matmul(X, W1)
    hidden_1 = z1 + ad.broadcast_to(b1, z1)
    activation_1 = au.nn.relu(hidden_1)
    z2 = ad.matmul(activation_1, W2)
    hidden_2 = z2 + ad.broadcast_to(b2, z2)
    activation_2 = au.nn.relu(hidden_2)
    z3 = ad.matmul(activation_2, W3)
    hidden_3 = z3 + ad.broadcast_to(b3, z3)
    loss = au.nn.softmax_cross_entropy_with_logits(hidden_3, y)
    return loss, W1, b1, W2, b2, W3, b3, hidden_3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--exe_context",
        help="Choose execution context: numpy, gpu",
        default="gpu",
    )
    parser.add_argument("-i", "--num_iter", help="Number of iterations", default=500)
    parser.add_argument(
        "--optim",
        help="Optimizer type: sgd or adam",
        default="adam",
        choices=["sgd", "adam"],
    )
    args = parser.parse_args()

    use_gpu = args.exe_context == "gpu"
    n_iter = int(args.num_iter)
    optim_type = args.optim

    start_time = timeit.default_timer()

    data = au.datasets.MNIST(batch_size=128)
    batch_generator = data.train_batch_generator()
    input_size = data.num_features()
    hid_1_size, hid_2_size, output_size = 256, 100, 10
    lr = 1e-3

    X = ad.Variable(name="X")
    y = ad.Variable(name="y")
    loss, W1, b1, W2, b2, W3, b3, logit = build_graph(
        X, y, input_size, hid_1_size, hid_2_size, output_size
    )
    params = [W1, b1, W2, b2, W3, b3]

    # ========== 三次保存图 ==========
    # 1. 前向图（从输入到损失和 logits）
    forward_end = [loss, logit]
    forward_start = [X, y] + params
    d2_fwd = to_d2lang(forward_end, forward_start, graph_name="MNIST_Forward")
    with open("mnist_forward.d2", "w") as f:
        f.write(d2_fwd)
    print("Forward graph saved to mnist_forward.d2")

    # 2. 反向图（前向 + 梯度节点，不包含优化器更新）
    grads = ad.gradients(loss, params)  # 梯度节点
    backward_end = [loss, logit] + grads
    backward_start = [X, y] + params
    d2_bwd = to_d2lang(backward_end, backward_start, graph_name="MNIST_Backward")
    with open("mnist_backward.d2", "w") as f:
        f.write(d2_bwd)
    print("Backward graph saved to mnist_backward.d2")

    # 3. 完整图（前向 + 反向 + 优化器更新）
    if optim_type == "sgd":
        new_params, new_vels, state_nodes = build_optimizer_updates(
            "sgd", loss, params, lr=lr, momentum=0.9
        )
        velocities = state_nodes[0]
        eval_list = [loss] + new_params + new_vels
        vel_vals = [xp.zeros_like(p.const, dtype=xp.float32) for p in params]
        state_vars = velocities
        state_vals = vel_vals
    else:  # adam
        new_params, new_moms, new_vels, state_nodes = build_optimizer_updates(
            "adam", loss, params, lr=lr, beta1=0.9, beta2=0.995, eps=1e-5
        )
        t_node, beta1_t_node, beta2_t_node, momentums, velocities = state_nodes
        eval_list = [loss] + new_params + new_moms + new_vels
        t_val = xp.array([1.0], dtype=xp.float32)
        beta1_t_val = xp.array([0.9**1.0], dtype=xp.float32)
        beta2_t_val = xp.array([0.995**1.0], dtype=xp.float32)
        mom_vals = [xp.zeros_like(p.const, dtype=xp.float32) for p in params]
        vel_vals = [xp.zeros_like(p.const, dtype=xp.float32) for p in params]
        state_vars = [t_node, beta1_t_node, beta2_t_node] + momentums + velocities
        state_vals = [t_val, beta1_t_val, beta2_t_val] + mom_vals + vel_vals

    full_end = [loss, logit] + new_params
    if optim_type == "adam":
        full_end += new_moms + new_vels
    else:
        full_end += new_vels
    full_start = [X, y] + params + state_vars
    d2_full = to_d2lang(
        full_end, full_start, graph_name=f"MNIST_{optim_type.upper()}_Full"
    )
    with open(f"mnist_full_{optim_type}.d2", "w") as f:
        f.write(d2_full)
    print(f"Full graph saved to mnist_full_{optim_type}.d2")

    # ========== 训练（使用完整图） ==========
    executor = ad.Executor(eval_list, use_gpu=use_gpu)
    n = len(params)
    param_vals = [p.const for p in params]

    for i in range(n_iter):
        X_batch, y_batch = next(batch_generator)
        feed = {X: X_batch, y: y_batch}
        for p, v in zip(params, param_vals):
            feed[p] = v
        for var, val in zip(state_vars, state_vals):
            feed[var] = xp.asarray(val, dtype=xp.float32)

        results = executor.run(feed)
        loss_now = results[0]
        new_param_vals = results[1 : 1 + n]
        param_vals = new_param_vals
        for p, v in zip(params, param_vals):
            p.const = v

        if optim_type == "sgd":
            new_vel_vals = results[1 + n : 1 + 2 * n]
            state_vals = new_vel_vals
        else:
            new_mom_vals = results[1 + n : 1 + 2 * n]
            new_vel_vals = results[1 + 2 * n : 1 + 3 * n]
            t_val = t_val + xp.array([1.0], dtype=xp.float32)
            beta1_t_val = xp.array([0.9 ** t_val[0]], dtype=xp.float32)
            beta2_t_val = xp.array([0.995 ** t_val[0]], dtype=xp.float32)
            state_vals = [t_val, beta1_t_val, beta2_t_val] + new_mom_vals + new_vel_vals

        if (
            i <= 10
            or (i <= 100 and i % 10 == 0)
            or (i <= 1000 and i % 100 == 0)
            or (i <= 10000 and i % 500 == 0)
        ):
            print(f"iter: {i:>5d} cost: {loss_now[0]:>8.5f}")

    val_acc = measure_accuracy(logit, data.validation(), use_gpu=use_gpu)
    test_acc = measure_accuracy(logit, data.testing(), use_gpu=use_gpu)
    print(f"Validation accuracy: {val_acc:.2f}%")
    print(f"Testing accuracy: {test_acc:.2f}%")

    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time:.3f} seconds")
