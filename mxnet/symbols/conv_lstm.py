# pylint:skip-file

import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias", ])
                                     # "c2h_weight", "c2h_bias",
                                     # "c_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, prefix='forward'):
    """LSTM Cell symbol"""
    i2h = mx.symbol.Convolution(
        data=indata,
        kernel=(5, 5),
        pad=(2, 2),
        num_filter=num_hidden * 4,
        num_group=4,
        weight=param.i2h_weight,
        bias=param.i2h_bias,
        name="{prefix}t{seq_id}_l{l_id}_i2h".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    )
    h2h = mx.symbol.Convolution(
        data=prev_state.h,
        kernel=(5, 5),
        pad=(2, 2),
        num_filter=num_hidden * 4,
        num_group=4,
        weight=param.h2h_weight,
        bias=param.h2h_bias,
        name="{prefix}t{seq_id}_l{l_id}_h2h".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    )

    # prev_context = mx.symbol.Concat(*[prev_state.c for _ in range(3)], dim=1)
    # c2h = param.c2h_weight * prev_context + param.c2h_bias
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(
        gates,
        num_outputs=4,
        name="{prefix}t{seq_id}_l{l_id}_slice".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    )

    # slice_context = mx.symbol.SliceChannel(
    #     c2h,
    #     num_outputs=3,
    #     name="{prefix}t{seq_id}_l{l_id}_slice_context".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    # )

    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    forget_gate = mx.sym.Activation(slice_gates[1], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def conv_lstm_unroll(data, layer_id, seq_len=1, num_hidden=256, blstm=False):

    forward_cell = LSTMParam(
        i2h_weight=mx.sym.Variable("l%d_forward_i2h_weight" % layer_id),
        i2h_bias=mx.sym.Variable("l%d_forward_i2h_bias" % layer_id),
        h2h_weight=mx.sym.Variable("l%d_forward_h2h_weight" % layer_id),
        h2h_bias=mx.sym.Variable("l%d_forward_h2h_bias" % layer_id),
        # c2h_weight=mx.symbol.Variable("l%d_forward_c2h_weight" % layer_id),
        # c2h_bias=mx.symbol.Variable("l%d_forward_c2h_bias" % layer_id),
        # c_bias=mx.symbol.Variable("l%d_forward_c_bias" % layer_id),
    )
    forward_state = LSTMState(
        c=mx.sym.Variable("l%d_forward_init_c_state_cell" % layer_id),
        h=mx.sym.Variable("l%d_forward_init_h_state" % layer_id)
    )

    if blstm:
        backward_cell = LSTMParam(
            i2h_weight=mx.sym.Variable("l%d_backward_i2h_weight" % layer_id),
            i2h_bias=mx.sym.Variable("l%d_backward_i2h_bias" % layer_id),
            h2h_weight=mx.sym.Variable("l%d_backward_h2h_weight" % layer_id),
            h2h_bias=mx.sym.Variable("l%d_backward_h2h_bias" % layer_id),
            # c2h_weight=mx.symbol.Variable("l%d_backward_c2h_weight" % layer_id),
            # c2h_bias=mx.symbol.Variable("l%d_backward_c2h_bias" % layer_id),
            # c_bias=mx.symbol.Variable("l%d_backward_c_bias" % layer_id),
        )
        backward_state = LSTMState(
            c=mx.sym.Variable("l%d_backward_init_c_state_cell" % layer_id),
            h=mx.sym.Variable("l%d_backward_init_h_state" % layer_id)
        )

    sliced = mx.symbol.SliceChannel(data=data, num_outputs=seq_len, axis=0, squeeze_axis=True)

    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = sliced[seqidx]
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=forward_state,
                          param=forward_cell,
                          seqidx=seqidx, layeridx=layer_id, prefix='forward')
        hidden = next_state.h
        forward_state = next_state
        forward_hidden.append(hidden)

    if blstm:
        backward_hidden = []
        for seqidx in reversed(range(seq_len)):
            hidden = sliced[seqidx]
            next_state = lstm(
                num_hidden,
                indata=hidden,
                prev_state=backward_state,
                param=backward_cell,
                seqidx=seqidx,
                layeridx=layer_id,
                prefix='backward',
            )
            hidden = next_state.h
            backward_state = next_state
            backward_hidden.append(hidden)
        backward_hidden = reversed(backward_hidden)

        layer_hidden = [mx.symbol.Concat(*[forward, backward], dim=1) for forward, backward in zip(forward_hidden, backward_hidden)]
    else:
        layer_hidden = forward_hidden

    hidden_concat = mx.sym.Concat(*[mx.symbol.expand_dims(hidden, axis=0) for hidden in layer_hidden], dim=0)

    if blstm:
        hidden_reshaped = mx.symbol.Reshape(data=hidden_concat, shape=(-1, num_hidden * 2))
        output_activations = mx.symbol.FullyConnected(data=hidden_reshaped, num_hidden=num_hidden, name="lstm_{}_output".format(layer_id))
        return mx.symbol.Reshape(data=output_activations, shape=(seq_len, -1, num_hidden))

    return hidden_concat
