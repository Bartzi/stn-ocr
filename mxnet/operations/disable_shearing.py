import os

import mxnet as mx

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"


class DisableShearing(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        out = in_data[0].copy().asnumpy().reshape(-1, 2, 3)
        out[:, 0, 1] = 0
        out[:, 1, 0] = 0
        self.assign(out_data[0], req[0], mx.nd.array(out.reshape(-1, 6)))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = out_grad[0].asnumpy().reshape(-1, 2, 3)
        grad[:, 0, 1] = 0
        grad[:, 1, 0] = 0
        self.assign(in_grad[0], req[0], mx.nd.array(grad.reshape(-1, 6)))


@mx.operator.register("DisableShearing")
class DisableShearingProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(DisableShearingProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0]], [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return DisableShearing()
