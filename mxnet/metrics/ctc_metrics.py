import logging
import math

import mxnet as mx
import numpy as np

from .base import STNAccuracy
from .ctc.ctc_loss import CTCLoss as C_CTCLoss


def remove_blank(label, blank_label=0):
    blank_indices = np.where(label == blank_label)[0]
    if len(blank_indices) == 0:
        return label

    return label[:blank_indices[0]]


def strip_prediction(prediction, blank_label=0):
    stripped_prediction = []
    enlarged_prediction = [blank_label] + list(prediction)

    for char_1, char_2 in zip(enlarged_prediction, prediction):
        if char_2 == blank_label or char_2 == char_1:
            continue
        stripped_prediction.append(char_2)

    return stripped_prediction


class CTCSTNAccuracy(STNAccuracy):

    def __init__(self, label_width, num_timesteps, blank_label=0):
        self.label_width = label_width
        self.num_timesteps = num_timesteps
        self.blank_label = blank_label
        super(CTCSTNAccuracy, self).__init__()

    def accuracy(self, labels, preds):
        labels = labels.reshape(self.label_width, -1)
        labels = labels.transpose(1, 0)
        _, num_classes = preds.shape
        preds = preds.reshape(self.num_timesteps, -1, num_classes)
        preds = np.transpose(preds, (1, 0, 2))

        data_iter = zip(
            np.split(labels, len(labels), axis=0),
            np.split(preds, len(preds), axis=0),
        )

        hits = 0
        for label, prediction in data_iter:
            label = np.squeeze(label, axis=0)
            prediction = np.squeeze(prediction, axis=0)

            label = remove_blank(label, blank_label=self.blank_label)
            pred = np.argmax(prediction, axis=1)
            pred = strip_prediction(pred, blank_label=self.blank_label)

            if len(label) == len(pred):
                match = True
                for l, p in zip(label, pred):
                    if l != p:
                        match = False
                        break
                if match:
                    hits += 1

        return hits, len(labels)


class CTCLoss(mx.metric.EvalMetric):

    def __init__(self, sequence_length, label_width, blank_label):
        super(CTCLoss, self).__init__('ctc_loss')
        self.sequence_length = sequence_length
        self.label_width = label_width
        self.loss_calculator = C_CTCLoss(sequence_length, label_width, blank_label)
        self.blank_label = blank_label

    def update(self, labels, preds):
        labels = labels[0].asnumpy().astype(np.int32)
        labels = labels.flatten()

        preds = preds[0].asnumpy().astype(np.float32)
        _, num_classes = preds.shape
        preds = np.ascontiguousarray(preds.reshape(self.sequence_length, -1, num_classes))

        loss = self.loss_calculator.calc_ctc_loss(labels, preds)
        if math.isnan(loss):
            logging.error(loss)

        self.sum_metric += loss
        self.num_inst += len(preds) // self.sequence_length
