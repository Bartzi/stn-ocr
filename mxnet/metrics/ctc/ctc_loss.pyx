cimport cctc_loss

import array
from cpython cimport array

import numpy as np
cimport numpy as np

FLOAT_TYPE = np.float32
ctypedef np.float32_t FLOAT_TYPE_t


INT_TYPE = np.int32
ctypedef np.int32_t INT_TYPE_t


cdef get_label_lengths(np.ndarray[INT_TYPE_t, ndim=1] labels, int batch_size, int blank, int label_width, int& total_length):
        cdef np.ndarray real_label_lengths = np.zeros([batch_size], dtype=INT_TYPE)
        cdef int i

        for i in range(len(labels)):
            if labels[i] == blank:
                continue
            real_label_lengths[i // label_width] += 1
            (&total_length)[0] += 1

        return real_label_lengths

cdef remove_blank(np.ndarray[INT_TYPE_t, ndim=1] labels, int blank, int total_label_length):
        cdef np.ndarray cpu_labels = np.zeros([total_label_length], dtype=INT_TYPE)
        cdef int k = 0
        cdef int i

        for i in range(len(labels)):
            if labels[i] != blank:
                cpu_labels[k] = labels[i]
                k += 1

        return cpu_labels


cdef class CTCLoss:

    cdef int sequence_length
    cdef int blank_label
    cdef int label_width

    def __init__(self, sequence_length, label_width, blank_label):
        pass

    def __cinit__(self, sequence_length, label_width, blank_label):
        self.sequence_length = sequence_length
        self.blank_label = blank_label
        self.label_width = label_width

    cpdef calc_ctc_loss(self, np.ndarray[INT_TYPE_t, ndim=1] labels, np.ndarray[FLOAT_TYPE_t, ndim=3] preds):
        # expected shapes:
        # labels: batch_size * label_width
        # preds: num_time_steps x batch_size x num_classes

        cdef int batch_size = preds.shape[1]
        cdef int alphabet_size = preds.shape[2]

        cdef np.ndarray[dtype=INT_TYPE_t, ndim=1] input_lengths = np.zeros([batch_size], dtype=INT_TYPE)
        cdef int i
        for i in range(batch_size):
            input_lengths[i] = self.sequence_length

        cdef int total_label_length = 0
        cdef np.ndarray[dtype=INT_TYPE_t, ndim=1] label_lengths = get_label_lengths(
            labels,
            batch_size,
            self.blank_label,
            self.label_width,
            total_label_length,
        )

        cdef np.ndarray[dtype=INT_TYPE_t, ndim=1] cpu_labels = remove_blank(
            labels,
            self.blank_label,
            total_label_length,
        )

        cdef cctc_loss.ctcOptions options
        options.loc = cctc_loss.CTC_CPU
        options.num_threads = 1
        options.blank_label = self.blank_label

        cdef size_t workspace_size
        cdef cctc_loss.ctcStatus_t ret
        ret = cctc_loss.get_workspace_size(
            <int*>&label_lengths[0],
            <int*>&input_lengths[0],
            alphabet_size,
            batch_size,
            options,
            &workspace_size,
        )

        cdef bytes reason
        if ret != cctc_loss.CTC_STATUS_SUCCESS:
            reason = <bytes> cctc_loss.ctcGetStatusString(ret)
            raise ValueError(reason)

        cdef array.array work_space = array.array('f', [0 for _ in range(workspace_size)])
        cdef np.ndarray[dtype=FLOAT_TYPE_t, ndim=1] costs = np.zeros([batch_size], dtype=FLOAT_TYPE)

        cdef np.ndarray[FLOAT_TYPE_t, ndim=3] grads = np.zeros_like(preds, dtype=FLOAT_TYPE)

        ret = cctc_loss.compute_ctc_loss(
            &preds[0, 0, 0],
            &grads[0, 0, 0],
            <int*>&cpu_labels[0],
            <int*>&label_lengths[0],
            <int*>&input_lengths[0],
            alphabet_size,
            batch_size,
            &costs[0],
            work_space.data.as_voidptr,
            options,
        )

        if ret != cctc_loss.CTC_STATUS_SUCCESS:
            reason = <bytes> cctc_loss.ctcGetStatusString(ret)
            raise ValueError(reason)

        cdef float total_cost = np.sum(costs)

        return total_cost

