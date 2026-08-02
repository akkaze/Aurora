import cunumpy as xp

from aurora.autodiff.autodiff import Op


class TanhOp(Op):
    """
    Tanh Activation function

    """

    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Tanh({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = xp.tanh(input_vals[0])


    def gradient(self, node, output_grads):
        x = node.inputs[0]
        g = 1 - (tanh(x) * tanh(x))
        return [g * output_grads]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes)
        return input_shapes[0]


class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f'exp({node_A.name})'
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = xp.exp(input_vals[0])


    def gradient(self, node, output_grads):
        return [exp(node.inputs[0]) * output_grads]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class LogOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f'log({node_A.name})'
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = xp.log(input_vals[0])

    def gradient(self, node, output_grads):
        return [output_grads / node.inputs[0]]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

class SqrtOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f'sqrt({node_A.name})'
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = xp.sqrt(input_vals[0])

    def gradient(self, node, output_grads):
        return [output_grads / (2 * sqrt(node.inputs[0]))]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

# Global singleton operations
tanh = TanhOp()
exp = ExpOp()
log = LogOp()
sqrt = SqrtOp()
# TODO: (upul) other basic math functions such as sin, cos, min, max, and etc
