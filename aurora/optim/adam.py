import cunumpy as xp
from .base import Base

try:
    from aurora.ndarray import gpu_op, ndarray
except ImportError:
    pass


class Adam(Base):
    def __init__(self, cost, params, lr=1e-3, beta1=0.9, beta2=0.995, eps=1e-5, use_gpu=False):
        super().__init__(cost, params, lr, use_gpu=use_gpu)
        self.beta1 = beta1
        self.beta2 = beta2

        self.velocity = [xp.zeros_like(param.const) for param in params]
        self.momentum = [xp.zeros_like(param.const) for param in params]

        self.time = 0
        self.eps = eps

    def step(self, feed_dict):
        exe_output = self.executor.run(feed_dict)
        self.time += 1

        vec_hat = [xp.zeros_like(param.const) for param in self.params]
        mom_hat = [xp.zeros_like(param.const) for param in self.params]

        for i in range(len(self.params)):
            self.momentum[i] = self.beta1 * self.momentum[i] + (1 - self.beta1) * exe_output[i + 1]
            mom_hat[i] = self.momentum[i] / (1 - self.beta1 ** self.time)

            self.velocity[i] = self.beta2 * self.velocity[i] + (1 - self.beta2) * (exe_output[i + 1] ** 2)
            vec_hat[i] = self.velocity[i] / (1 - self.beta2 ** self.time)

        for i in range(len(self.params)):
            self.params[i].const += -self.lr * mom_hat[i] / (xp.sqrt(vec_hat[i]) + self.eps)

        cost = exe_output[0]
        return cost
