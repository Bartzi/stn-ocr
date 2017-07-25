import argparse
import logging
import os

import mxnet as mx
from callbacks.create_checkpoint import get_create_checkpoint_callback


def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('train_file', type=str, help='the csv file containing all training data')
    parser.add_argument('val_file', type=str, help='the csv file containing all validation data')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.001,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    parser.add_argument('--plot', action='store_true', default=False, dest='plot_network_graph',
                        help='plot the computational graph of the network to a file')
    parser.add_argument('--log-dir', '-l', default='logs', help='path to base log directory [default: logs]')
    parser.add_argument('--log-name', '-ln', default='training', help='name of the directory where logs and plots shall be put in [default: training]')
    parser.add_argument('--log-level', default='INFO', help='sets the log level [default: INFO]')
    parser.add_argument('--send-bboxes', default=False, action='store_true', help='send bbox images for each time step to upstream viewer (you must provide an upstream if you are using this feature)')
    parser.add_argument('--ip', help='upstream ip that can recieve bboxes')
    parser.add_argument('--port', default=1337, type=int, help='remote port to connect to')
    parser.add_argument('--gif', action='store_true', default=False, help='create a gif of plotted bboxes')
    parser.add_argument('--video', action='store_true', default=False, help='create a video of plotted bboxes')
    parser.add_argument('--fix-loc', action='store_true', default=False, help='fix the params of the localisation network')
    parser.add_argument('--progressbar', action='store_true', default=False, help='show a progressbar')
    parser.add_argument('--char-map', help='path to char map in json format, used to display current prediction')
    parser.add_argument('--blank-label', type=int, default=0, help='label that indicates the blank case [default: 0]')
    parser.add_argument('-ci', '--checkpoint_interval', type=int, help='number of iterations after which a checkpoint shall be saved')
    parser.add_argument('--zoom', type=float, default=0.9, help='default zoom value for initialisation of STN param predictor')
    parser.add_argument('--eval-image', help='path to image that shall be used by bbox plotter [default: take random image from val dataset]')
    return parser


def init_logging(args, kv, epoch_size):
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    logger = logging.getLogger()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(args.log_level.upper())
    logger.info('start with arguments %s', args)
    logger.info('EPOCH SIZE: %s', int(epoch_size))


def fit(args, network, train_data, epoch_size, initializer, eval_metric, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    init_logging(args, kv, epoch_size)

    # load model
    model_prefix = args.model_prefix
    # if model_prefix is not None:
    #     model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params': tmp.arg_params,
                      'aux_params': tmp.aux_params,
                      'begin_epoch': args.load_epoch}
        # TODO: check epoch_size for 'dist_sync'
        model_args['begin_num_update'] = epoch_size * args.load_epoch

    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    if save_model_prefix is not None:
        save_model_prefix = os.path.join(args.log_dir, 'models', save_model_prefix)
        os.makedirs(os.path.dirname(save_model_prefix), exist_ok=True)
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    train, val = train_data

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    lr_scheduler = None
    if 'lr_factor' in args and args.lr_factor < 1:
        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            step=max(int(epoch_size * args.lr_factor_epoch), 1),
            factor=args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
                    args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx=devs,
        symbol=network,
        num_epoch=args.num_epochs,
        epoch_size=epoch_size,
        learning_rate=args.lr,
        # optimizer=mx.optimizer.AdaDelta(),
        # optimizer=mx.optimizer.Adam(learning_rate=args.lr),
        optimizer=mx.optimizer.SGD(momentum=0.9, learning_rate=args.lr, lr_scheduler=lr_scheduler),
        # optimizer=mx.optimizer.RMSProp(learning_rate=args.lr, lr_scheduler=lr_scheduler, clip_gradient=5),
        momentum=0.9,
        wd=0.00001,
        # initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        initializer=initializer,
        **model_args)

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

    if args.checkpoint_interval is not None:
        batch_end_callback.append(get_create_checkpoint_callback(args.checkpoint_interval, save_model_prefix))

    if args.progressbar:
        batch_end_callback.append(mx.callback.ProgressBar(epoch_size))

    model.fit(
        X=train,
        eval_data=val,
        eval_metric=eval_metric,
        kvstore=kv,
        # monitor=monitor,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=checkpoint)
