from collections import namedtuple

import mxnet as mx
from symbols.lstm import LSTMState, LSTMParam, lstm

AttentionLSTMParam = namedtuple(
    "LSTMParam",
    [
        "i2h_weight",
        "i2h_bias",
        "h2h_weight",
        "h2h_bias",
        "c2h_weight",
        "c2h_bias",
    ]
)


class Attention:

    def __init__(self, context_shape, state_dim):
        self.w_v = mx.symbol.Variable('energy_V_weight', shape=(context_shape[2], context_shape[1]))
        self.w_g = mx.symbol.Variable('energy_W_weight', shape=(context_shape[2], context_shape[1]))
        self.w_h = mx.symbol.Variable('energy_H_weight', shape=(1, context_shape[2]))
        # self.ones = mx.symbol.Variable('energy_ones', shape=(1, context_shape[2]))
        one_dummy = mx.symbol.Variable('ones_dummy_bias', shape=(1, context_shape[2]))
        self.ones = mx.symbol.Custom(one_dummy, op_type="ProvideOnes")

        self.context_shape = context_shape
        self.state_dim = state_dim

    def attend(self, context, state):
        context_slices = mx.symbol.SliceChannel(context, num_outputs=self.context_shape[0], axis=0, squeeze_axis=True)
        state_slices = mx.symbol.SliceChannel(state, num_outputs=self.context_shape[0], axis=0, squeeze_axis=True)

        alphas = []
        contexts = []
        for i in range(self.context_shape[0]):
            context_slice = context_slices[i]
            state_slice = state_slices[i]
            state_slice = mx.symbol.expand_dims(state_slice, axis=1)
            projected_state = mx.symbol.dot(self.w_g, state_slice)
            projected_state = mx.symbol.dot(projected_state, self.ones)

            projected_context = mx.symbol.dot(self.w_v, context_slice)
            projected_context = projected_context + projected_state
            projected_context = mx.symbol.Activation(projected_context, act_type='tanh')
            projected_context = mx.symbol.dot(self.w_h, projected_context)
            alpha = mx.symbol.SoftmaxActivation(projected_context)
            alphas.append(alpha)
            context_slice_transposed = mx.symbol.transpose(context_slice)
            weighted_context = mx.symbol.dot(alpha, context_slice_transposed)
            contexts.append(weighted_context)

        weighted_contexts = mx.symbol.Concat(*contexts, dim=0)
        alphas = mx.symbol.Concat(*alphas, dim=0)
        return weighted_contexts, alphas


def attention_lstm(num_hidden, indata, prev_state, context, param, seqidx, layeridx, prefix='forward'):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(
        data=indata,
        weight=param.i2h_weight,
        bias=param.i2h_bias,
        num_hidden=num_hidden * 4,
        name="{prefix}t{seq_id}_l{l_id}_i2h".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    )
    h2h = mx.sym.FullyConnected(
        data=prev_state.h,
        weight=param.h2h_weight,
        bias=param.h2h_bias,
        num_hidden=num_hidden * 4,
        name="{prefix}t{seq_id}_l{l_id}_h2h".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    )
    c2h = mx.symbol.FullyConnected(
        data=context,
        weight=param.c2h_weight,
        bias=param.c2h_bias,
        num_hidden=num_hidden * 4,
        name="{prefix}t{seq_id}_l{l_id}_c2h".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    )
    gates = i2h + h2h + c2h
    slice_gates = mx.sym.SliceChannel(
        gates,
        num_outputs=4,
        name="{prefix}t{seq_id}_l{l_id}_slice".format(prefix=prefix, seq_id=seqidx, l_id=layeridx)
    )

    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def attention_lstm_unroll(data, context, attention, layer_id, seq_len=1, num_hidden=256):
    forward_cell = LSTMParam(
        i2h_weight=mx.symbol.Variable("l%d_forward_i2h_weight" % layer_id),
        i2h_bias=mx.symbol.Variable("l%d_forward_i2h_bias" % layer_id),
        h2h_weight=mx.symbol.Variable("l%d_forward_h2h_weight" % layer_id),
        h2h_bias=mx.symbol.Variable("l%d_forward_h2h_bias" % layer_id),
    )

    forward_state = LSTMState(
        c=mx.symbol.Variable("l{}_forward_init_c_state_cell".format(layer_id)),
        h=mx.symbol.Variable("l{}_forward_init_h_state".format(layer_id)),
    )

    sliced = mx.symbol.SliceChannel(data=data, num_outputs=seq_len, axis=0, squeeze_axis=True)

    outputs = []
    alphas = []
    for seq_idx in range(seq_len):
        hidden = sliced[seq_idx]

        next_state = lstm(attention.context_shape[1], indata=hidden,
                          prev_state=forward_state,
                          param=forward_cell,
                          seqidx=seq_idx, layeridx=layer_id, prefix='forward')

        hidden = next_state.h
        forward_state = next_state

        weighted_context, alpha = attention.attend(context, forward_state.h)
        # alpha = mx.symbol.Custom(alpha, op_type="DebugOp")
        alphas.append(alpha)

        output_input = mx.symbol.Concat(hidden, weighted_context, dim=1)
        # output_input = hidden * weighted_context
        output = mx.symbol.FullyConnected(output_input, num_hidden=num_hidden)
        outputs.append(output)

    output_concat = mx.symbol.Concat(*[mx.symbol.expand_dims(output, axis=0) for output in outputs], dim=0)
    alpha_concat = mx.symbol.Concat(*[mx.symbol.expand_dims(alpha, axis=0) for alpha in alphas], dim=0)

    return output_concat, alpha_concat
