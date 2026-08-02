import aurora.autodiff as ad


class Base:
    def __init__(self, cost, params, lr=0.1, use_gpu=False):
        self.cost = cost

        # if use_gpu == True, create matrices in GPU
        self.params = params
        self.lr = lr
        grads = ad.gradients(cost, params)
        grads.insert(0, cost)
        self.use_gpu = use_gpu
        self.executor = ad.Executor(grads, use_gpu=use_gpu)

    def step(self, feed_dict):
        raise NotImplementedError('This method should be implemented by subclasses')
