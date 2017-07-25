import mxnet as mx
import numpy as np

from .base import STNAccuracy


def tile_batch_size(array, size=4):
    array = array[:, np.newaxis, ...]
    array = np.tile(array, (1, size, 1))
    array_shape = array.shape
    return array.reshape(-1, *array_shape[2:])


class FSNSPretrainAccuracy(STNAccuracy):

    def update(self, labels, preds):
        cls_prob = preds[0].asnumpy()
        labels = labels[0].asnumpy()

        # copy the same labels 4 times because we have 4 times the same image (at least we assume that)
        labels = tile_batch_size(labels)

        accuracy_sum, num_instances = self.accuracy(labels, cls_prob)
        self.sum_metric += accuracy_sum
        self.num_inst += num_instances


class FSNSPretrainCrossEntropy(mx.metric.EvalMetric):

    def __init__(self, eps=1e-8):
        super(FSNSPretrainCrossEntropy, self).__init__('cross-entropy')
        self.eps = eps

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = tile_batch_size(label)
            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]



