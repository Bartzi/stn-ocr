import mxnet as mx

from networks.text_rec import LocalizationNetwork
from symbols.lstm import lstm_unroll


class SVHNLocalizationNetwork:

    @staticmethod
    def get_network(data, source_shape, num_timesteps, num_rnn_layers=1, blstm=False, attr={'lr_mult': '0.01'}):

        h = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=32, name="loc_conv0")
        h = mx.symbol.BatchNorm(data=h)
        h = mx.symbol.Activation(data=h, act_type="relu")
        pre_res_net = mx.symbol.Pooling(data=h, pool_type="avg", kernel=(2, 2), stride=(2, 2))

        h = mx.symbol.Convolution(data=pre_res_net, kernel=(3, 3), pad=(1, 1), num_filter=32, name="loc_conv1_1")
        h = mx.symbol.BatchNorm(data=h)
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=32, name="loc_conv1_2")
        h = mx.symbol.BatchNorm(data=h)
        h_conv_1 = h + pre_res_net
        h_conv_1 = mx.symbol.Activation(data=h_conv_1, act_type="relu")

        h_pool = mx.symbol.Pooling(data=h_conv_1, kernel=(2, 2), stride=(2, 2), pool_type='max')

        h = mx.symbol.Convolution(data=h_pool, kernel=(3, 3), pad=(1, 1), num_filter=48, name="loc_conv2_1")
        h = mx.symbol.BatchNorm(data=h)
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=48, name="loc_conv2_2")
        h_pre_short_2 = mx.symbol.Convolution(h_pool, kernel=(1, 1), num_filter=48)
        h_pre_short_2 = mx.symbol.BatchNorm(data=h_pre_short_2)
        h_conv_2 = h + h_pre_short_2
        h_conv_2 = mx.symbol.Activation(data=h_conv_2, act_type="relu")

        h_conv_2 = mx.symbol.Pooling(data=h_conv_2, kernel=(2, 2), stride=(2, 2), pool_type='max')

        h = mx.symbol.Convolution(data=h_conv_2, kernel=(3, 3), pad=(1, 1), num_filter=48, name="loc_conv3_1")
        h = mx.symbol.BatchNorm(data=h)
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=48, name="loc_conv3_2")
        h = mx.symbol.BatchNorm(data=h)
        h = h + h_conv_2

        h = mx.symbol.Pooling(h, pool_type='avg', kernel=(10, 10))
        h = mx.symbol.Flatten(data=h)

        size_params = next(iter(mx.symbol_doc.SymbolDoc.get_output_shape(h, data=source_shape).values()))

        pre_concat = mx.symbol.expand_dims(h, axis=0)
        lstm_input = mx.symbol.Concat(*[pre_concat for _ in range(num_timesteps)], dim=0, name="prepare_lstm")

        for i in range(num_rnn_layers):
            rnn = lstm_unroll(lstm_input, layer_id=i, seq_len=num_timesteps, num_hidden=256, blstm=blstm)
            lstm_input = rnn

        # rnn = mx.symbol.Activation(data=rnn, act_type='relu')
        rnn_flat = mx.symbol.Reshape(data=rnn, shape=(-3, -1))
        loc = mx.symbol.FullyConnected(data=rnn_flat, num_hidden=6, name="stn_loc", attr=attr)

        return loc, size_params


class SVHNMultiLineResNetNetwork:

    @staticmethod
    def get_network(source_shape, target_shape, num_timesteps, num_rnn_layers, num_labels, blstm=False, fix_loc=False):
        data = mx.symbol.Variable('data')

        loc, size_params = SVHNLocalizationNetwork.get_network(
            data,
            source_shape,
            num_timesteps,
            num_rnn_layers=num_rnn_layers,
            blstm=blstm,
        )
        concat_data = mx.symbol.Concat(*[data for _ in range(num_timesteps)], dim=0, name="concat_input_for_stn")
        transformed = mx.symbol.SpatialTransformer(data=concat_data, loc=loc,
                                                   target_shape=(target_shape.height, target_shape.width),
                                                   transform_type="affine", sampler_type="bilinear", name='stn')
        if fix_loc:
            transformed = mx.symbol.BlockGrad(transformed)

        h = mx.symbol.Convolution(data=transformed, kernel=(3, 3), pad=(1, 1), num_filter=32, name="rec_conv0")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_0')
        h = mx.symbol.Activation(data=h, act_type="relu")
        pre_res_net = mx.symbol.Pooling(data=h, pool_type="avg", kernel=(2, 2), stride=(2, 2))

        h = mx.symbol.Convolution(data=pre_res_net, kernel=(3, 3), pad=(1, 1), num_filter=32, name="rec_conv1_1")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_1')
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=32, name="rec_conv1_2")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_2')
        h_conv_1 = h + pre_res_net

        h = mx.symbol.Convolution(data=h_conv_1, kernel=(3, 3), pad=(1, 1), num_filter=48, name="rec_conv2_1")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_3')
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=48, name="rec_conv2_2")
        h_pre_short_2 = mx.symbol.Convolution(h_conv_1, kernel=(1, 1), num_filter=48)
        h_pre_short_2 = mx.symbol.BatchNorm(data=h_pre_short_2, name='rec_bn_4')
        h_conv_2 = h + h_pre_short_2

        h = mx.symbol.Convolution(data=h_conv_2, kernel=(3, 3), pad=(1, 1), num_filter=48, name="rec_conv3_1")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_5')
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=48, name="rec_conv3_2")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_6')
        h = h + h_conv_2

        h = mx.symbol.Pooling(h, pool_type='avg', kernel=(10, 10))
        flat_h = mx.symbol.Flatten(data=h)
        h = mx.symbol.FullyConnected(data=flat_h, num_hidden=256, name='rec_fn_0')
        h = mx.symbol.Activation(data=h, act_type="relu")

        classifiers = []
        for i in range(num_labels):
            softmax = mx.symbol.FullyConnected(data=h, num_hidden=11, name='rec_softmax_{}'.format(i))
            softmax = mx.symbol.Reshape(softmax, shape=(num_timesteps, -1, 11))
            softmax = mx.symbol.expand_dims(softmax, axis=1)
            classifiers.append(softmax)

        h = mx.symbol.Concat(*classifiers, dim=1, name="concat_softmax_output")
        h = mx.symbol.Reshape(h, shape=(-1, 11))

        stored_label = mx.symbol.Variable('softmax_label')
        label = mx.symbol.SwapAxis(data=stored_label, dim1=1, dim2=0)
        flat_label = mx.symbol.Reshape(data=label, shape=(-1,))
        lenet = mx.symbol.SoftmaxOutput(data=h, label=flat_label, name='softmax')

        return lenet, loc, transformed, size_params


class SVHNMultiLineCTCNetwork:

    @staticmethod
    def get_network(source_shape, target_shape, num_timesteps, num_rnn_layers, num_labels, blstm=False, fix_loc=False):
        data = mx.symbol.Variable('data')

        loc, size_params = LocalizationNetwork.get_network(
            data,
            source_shape,
            num_timesteps,
            num_rnn_layers=num_rnn_layers - 1,
            blstm=blstm,
        )

        concat_data = mx.symbol.Concat(*[data for _ in range(num_timesteps)], dim=0, name="concat_input_for_stn")
        transformed = mx.symbol.SpatialTransformer(data=concat_data, loc=loc,
                                                   target_shape=(target_shape.height, target_shape.width),
                                                   transform_type="affine", sampler_type="bilinear", name='stn')
        if fix_loc:
            transformed = mx.symbol.BlockGrad(transformed)

        h = mx.symbol.Convolution(data=transformed, kernel=(3, 3), pad=(1, 1), num_filter=32, name="rec_conv0")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_0')
        h = mx.symbol.Activation(data=h, act_type="relu")
        pre_res_net = mx.symbol.Pooling(data=h, pool_type="avg", kernel=(2, 2), stride=(2, 2))

        h = mx.symbol.Convolution(data=pre_res_net, kernel=(3, 3), pad=(1, 1), num_filter=32, name="rec_conv1_1")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_1')
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=32, name="rec_conv1_2")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_2')
        h_conv_1 = h + pre_res_net

        h = mx.symbol.Convolution(data=h_conv_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="rec_conv2_1")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_3')
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=64, name="rec_conv2_2")
        h_pre_short_2 = mx.symbol.Convolution(h_conv_1, kernel=(1, 1), num_filter=64)
        h_pre_short_2 = mx.symbol.BatchNorm(data=h_pre_short_2, name='rec_bn_4')
        h_conv_2 = h + h_pre_short_2
        h_conv_2 = mx.symbol.Pooling(data=h_conv_2, pool_type="max", kernel=(2, 2), stride=(2, 2))

        h = mx.symbol.Convolution(data=h_conv_2, kernel=(3, 3), pad=(1, 1), num_filter=128, name="rec_conv3_1")
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_5')
        h = mx.symbol.Activation(data=h, act_type="relu")
        h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=128, name="rec_conv3_2")
        h_pre_short_3 = mx.symbol.Convolution(h_conv_2, kernel=(1, 1), num_filter=128)
        h_pre_short_3 = mx.symbol.BatchNorm(data=h_pre_short_3, name='rec_bn_7')
        h = mx.symbol.BatchNorm(data=h, name='rec_bn_6')
        h = h + h_pre_short_3

        h = mx.symbol.Pooling(h, pool_type='avg', kernel=(5, 5))
        flat_h = mx.symbol.Flatten(data=h)
        h = mx.symbol.FullyConnected(data=flat_h, num_hidden=256, name='rec_fn_0')
        h = mx.symbol.Activation(data=h, act_type="relu")

        h = mx.symbol.Reshape(h, shape=(num_timesteps, -1, 256))
        rnn = lstm_unroll(h, layer_id=num_rnn_layers - 1, seq_len=num_timesteps, num_hidden=256, blstm=blstm)
        rnn = mx.symbol.Reshape(rnn, shape=(-1, 256))

        softmax = mx.symbol.FullyConnected(data=rnn, num_hidden=11, name='rec_softmax')
        stored_label = mx.symbol.Variable('softmax_label')
        flat_label = mx.symbol.Reshape(data=stored_label, shape=(-1,))
        flat_label = mx.symbol.Cast(data=flat_label, dtype='int32')
        loss = mx.symbol.WarpCTC(data=softmax, label=flat_label, label_length=num_labels, input_length=num_timesteps)

        return loss, loc, transformed, size_params