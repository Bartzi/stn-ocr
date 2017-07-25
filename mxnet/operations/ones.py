import os

import mxnet as mx

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"


class ProvideOnes(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.ones(in_data[0].shape))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], mx.nd.zeros(out_grad[0].shape))


@mx.operator.register("ProvideOnes")
class ProvideOnesProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(ProvideOnesProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0]], [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return ProvideOnes()
