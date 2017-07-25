import datetime
import os

import mxnet as mx

import train_utils as utils
from data_io.file_iter import FileBasedIter
from initializers.spn_initializer import SPNInitializer
from utils.datatypes import Size
from utils.plot_log import LogPlotter


def get_network():
    data = mx.symbol.Variable("data")

    h = mx.symbol.Pooling(data=data, pool_type="avg", kernel=(2, 2), stride=(2, 2))
    h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=32)
    h = mx.symbol.BatchNorm(data=h)
    h = mx.symbol.Activation(data=h, act_type="relu")
    h = mx.symbol.Pooling(data=h, pool_type="max", kernel=(2, 2), stride=(2, 2))

    h = mx.symbol.Convolution(data=h, kernel=(3, 3), pad=(1, 1), num_filter=64)
    h = mx.symbol.BatchNorm(data=h)
    h = mx.symbol.Activation(data=h, act_type="relu")
    h = mx.symbol.Pooling(data=h, pool_type="max", kernel=(2, 2), stride=(2, 2))

    h = mx.symbol.FullyConnected(data=h, num_hidden=256)
    h = mx.symbol.Activation(data=h, act_type="relu")
    h = mx.symbol.FullyConnected(data=h, num_hidden=2)

    label = mx.symbol.Variable("softmax_label")
    label = mx.symbol.Reshape(label, shape=(-1,))
    loss = mx.symbol.SoftmaxOutput(data=h, label=label, name='softmax')

    return loss


if __name__ == "__main__":
    parser = utils.parse_args()
    args = parser.parse_args()

    time = datetime.datetime.now().isoformat()
    args.log_dir = os.path.join(args.log_dir, "{}_{}".format(time, args.log_name))
    args.log_file = os.path.join(args.log_dir, 'log')

    image_size = Size(width=150, height=150)
    source_shape = (args.batch_size, 3, image_size.height, image_size.width)

    eval_metric = mx.metric.CompositeEvalMetric(
        metrics=[
            mx.metric.create('ce'),
            mx.metric.create('accuracy'),
        ]
    )

    network = get_network()

    train_iter = FileBasedIter(
        args.train_file,
        args.batch_size,
        1,
        resize_to=image_size,
        base_dir='/',
        delimiter='\t',
        image_mode='RGB',
    )

    num_images = train_iter.num_data

    val_iter = FileBasedIter(
        args.val_file,
        args.batch_size,
        1,
        resize_to=image_size,
        base_dir='/',
        delimiter='\t',
        image_mode='RGB',
    )

    initializer = SPNInitializer(zoom=1)
    utils.fit(args, network, (train_iter, val_iter), num_images / args.batch_size, initializer, eval_metric, batch_end_callback=[])

    if hasattr(train_iter, 'shutdown'):
        train_iter.shutdown()
    if hasattr(val_iter, 'shutdown'):
        val_iter.shutdown()

    log_plotter = LogPlotter(args.log_file)
    plot = log_plotter.plot()
    plot.savefig(os.path.join(args.log_dir, 'plot.png'))