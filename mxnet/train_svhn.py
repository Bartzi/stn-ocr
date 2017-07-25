# import os
# os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
import datetime

import train_utils as utils
from callbacks.save_bboxes import BBOXPlotter
from data_io.file_iter import FileBasedIter
from data_io.lstm_iter import LSTMIter, InitStateLSTMIter
from initializers.spn_initializer import SPNInitializer, ShapeAgnosticLoad
from metrics.base import STNCrossEntropy, STNAccuracy
from networks.svhn import SVHNMultiLineResNetNetwork
from utils.create_gif import make_gif
from utils.create_video import make_video
from utils.datatypes import Size

from operations.debug import *
from operations.ones import *
from operations.disable_shearing import *
from utils.plot_log import LogPlotter



if __name__ == '__main__':
    parser = utils.parse_args()
    args = parser.parse_args()

    if args.send_bboxes and args.ip is None:
        parser.print_usage()
        raise ValueError("You must specify an upstream ip if you want to send the bboxes of each iteration")

    time = datetime.datetime.now().isoformat()
    args.log_dir = os.path.join(args.log_dir, "{}_{}".format(time, args.log_name))
    args.log_file = os.path.join(args.log_dir, 'log')

    image_size = Size(width=200, height=200)
    source_shape = (args.batch_size, 1, image_size.height, image_size.width)
    target_shape = Size(width=50, height=50)

    # adjustable network parameters
    num_timesteps = 1
    labels_per_timestep = 2
    num_rnn_layers = 1
    label_width = num_timesteps * labels_per_timestep
    use_blstm = False

    net, loc, transformed_output, size_params = SVHNMultiLineResNetNetwork.get_network(
        source_shape,
        target_shape,
        num_timesteps,
        num_rnn_layers,
        labels_per_timestep,
        blstm=use_blstm,
        fix_loc=args.fix_loc,
    )

    group = mx.symbol.Group([loc, transformed_output, net])

    if args.plot_network_graph:
        print("producing graph")
        graph = mx.visualization.plot_network(net)
        graph.render(os.path.join(args.log_dir, "graph.pdf"))
        print("rendered graph")

    print("loading data")

    train_iter = InitStateLSTMIter(
        base_iter=FileBasedIter(
            args.train_file,
            args.batch_size,
            label_width,
            resize_to=image_size,
            base_dir='/',
            delimiter='\t',
            image_mode='RGB',
        ),
        num_lstm_layers=num_rnn_layers,
        blstm=use_blstm,
        state_size=256,
    )

    num_images = train_iter.num_data

    val_iter = InitStateLSTMIter(
        base_iter=FileBasedIter(
            args.val_file,
            args.batch_size,
            label_width,
            resize_to=image_size,
            base_dir='/',
            delimiter='\t',
            image_mode='RGB',
        ),
        num_lstm_layers=num_rnn_layers,
        blstm=use_blstm,
        state_size=256,
    )
    
    iterations_per_epoch = num_images // args.batch_size
    num_iterations = iterations_per_epoch * args.num_epochs
    first_batch = next(iter(val_iter))
    val_iter.reset()
    bbox_data = first_batch.data[0]
    bbox_data = bbox_data.asnumpy()[1][np.newaxis, ...]
    bbox_label = first_batch.label[0][1].asnumpy()
    bbox_plotter = BBOXPlotter(
        image_size,
        target_shape,
        args.log_dir,
        save_labels=False,
        plot_extra_loc=False,
        send_bboxes=args.send_bboxes,
        upstream_ip=args.ip,
        upstream_port=args.port,
    )

    eval_metric = mx.metric.CompositeEvalMetric(
        metrics=[
            STNAccuracy(),
            STNCrossEntropy(),
        ]
    )

    callbacks = [bbox_plotter.get_callback(group, bbox_data, bbox_label, num_data=num_images, batch_num=1)]

    initializer = SPNInitializer(factor_type="in", magnitude=2.34, zoom=args.zoom)
    if args.model_prefix is not None:
        initializer = ShapeAgnosticLoad(args.model_prefix, default_init=initializer, verbose=True)
    # train
    utils.fit(args, net, (train_iter, val_iter), num_images / args.batch_size, initializer, eval_metric, batch_end_callback=callbacks)

    log_plotter = LogPlotter(args.log_file)
    plot = log_plotter.plot()
    plot.savefig(os.path.join(args.log_dir, 'plot.png'))

    if args.gif:
        make_gif(
            bbox_plotter.bbox_dir,
            os.path.join(bbox_plotter.base_dir, "bboxes.gif"),
            image_stride=num_iterations // min(num_iterations, 2000)
        )

    if args.video:
        make_video(
            bbox_plotter.bbox_dir,
            os.path.join(bbox_plotter.base_dir, "bboxes.mpeg"),
        )
