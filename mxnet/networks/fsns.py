import mxnet as mx
from networks.text_rec import LocalizationNetwork
from symbols.lstm import lstm_unroll


class FSNSNetwork:

    @staticmethod
    def get_network(source_shape, target_shape, num_timesteps, num_rnn_layers, num_labels, blstm=False, fix_loc=False):
        data = mx.symbol.Variable("data")
        data = mx.symbol.Reshape(data, shape=(-3, 0, 0, 0))

        loc, size_params = LocalizationNetwork.get_network(
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

        # merge extracted features of all 4 images together to (hopefully) get better recognition accuracies
        h = mx.symbol.Reshape(data=h, shape=(-4, -1, 4, -2))
        h = mx.symbol.Reshape(data=h, shape=(0, -3, -2))
        h = mx.symbol.Flatten(data=h)
        h = mx.symbol.FullyConnected(data=h, num_hidden=256, name="rec_fn_0")
        h = mx.symbol.Activation(data=h, act_type="relu")

        classifiers = []
        for i in range(num_labels):
            softmax = mx.symbol.FullyConnected(data=h, num_hidden=134, name='rec_softmax_{}'.format(i))
            softmax = mx.symbol.Reshape(softmax, shape=(num_timesteps, -1, 134))
            softmax = mx.symbol.expand_dims(softmax, axis=1)
            classifiers.append(softmax)

        h = mx.symbol.Concat(*classifiers, dim=1, name="concat_softmax_output")
        h = mx.symbol.Reshape(h, shape=(-1, 134))

        stored_label = mx.symbol.Variable('softmax_label')
        stored_label = mx.symbol.SwapAxis(data=stored_label, dim1=1, dim2=0)

        flat_label = mx.symbol.Reshape(data=stored_label, shape=(-1,))
        loss = mx.symbol.SoftmaxOutput(data=h, label=flat_label, name="softmax")

        return loss, loc, transformed, size_params
