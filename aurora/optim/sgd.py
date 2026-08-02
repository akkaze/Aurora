import cunumpy as xp
from .base import Base
try:
    from aurora.ndarray import gpu_op, ndarray
except ImportError:
    pass


class SGD(Base):
    def __init__(self, cost, params, lr=0.1, momentum=0.9, use_gpu=False):
        super().__init__(cost, params, lr=lr, use_gpu=use_gpu)
        self.momentum = momentum
        self.velocity = [xp.zeros_like(param.const) for param in params]

    def step(self, feed_dict):
        exe_output = self.executor.run(feed_dict)
        for i in range(len(self.params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * exe_output[1 + i]
            self.params[i].const += self.velocity[i]

        cost = exe_output[0]
        return cost
