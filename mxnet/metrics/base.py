import mxnet as mx
import numpy as np

from utils.bbox_utils import get_sampling_grid


class STNAccuracy(mx.metric.EvalMetric):

    def __init__(self, make_label_time_major=True):
        super(STNAccuracy, self).__init__('Accuracy')
        self.make_label_time_major = make_label_time_major

    def accuracy(self, labels, preds):
        if preds.shape != labels.shape:
            preds = np.argmax(preds, axis=1)
        preds = preds.astype(np.int32)
        labels = labels.astype(np.int32)

        sum_metric = (preds.flat == labels.flat).sum()
        return sum_metric, len(preds.flat)

    def update(self, labels, preds):
        cls_prob = preds[0].asnumpy()
        labels = labels[0].asnumpy()
        cls_labels = labels

        if len(cls_labels.shape) == 2:
            # make labels time major
            if self.make_label_time_major:
                cls_labels = np.transpose(cls_labels, axes=(1, 0))
            cls_labels = cls_labels.ravel()

        accuracy_sum, num_instances = self.accuracy(cls_labels, cls_prob)
        self.sum_metric += accuracy_sum
        self.num_inst += num_instances


class STNCrossEntropy(mx.metric.EvalMetric):

    def __init__(self, eps=1e-8, make_label_time_major=True):
        super(STNCrossEntropy, self).__init__('Loss')
        self.eps = eps
        self.make_label_time_major = make_label_time_major

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 2 and self.make_label_time_major:
                # make label time major
                label = np.transpose(label, axes=(1, 0))

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]
