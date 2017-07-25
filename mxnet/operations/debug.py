import logging
import os

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"

import mxnet as mx
import numpy as np


class Debug(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        nan = np.isnan(x)
        num_nan = nan[nan == True]
        logging.log(logging.DEBUG, "Forward: max: {}, mean: {}, min: {}, nan: {}".format(x.max(), x.mean(), x.min(), len(num_nan) / len(x.flatten())))
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = in_grad[0].asnumpy()
        nan = np.isnan(grad)
        num_nan = nan[nan==True]
        logging.log(logging.DEBUG, "Backward: min: {}, mean: {}, max: {} nan: {}".format(grad.min(), grad.mean(), grad.max(), len(num_nan) / len(grad.flatten())))
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("Debug")
class DebugProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(DebugProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def create_operator(self, ctx, shapes, dtypes):
        return Debug()
