import argparse
import csv
import json
import os
from collections import namedtuple

from PIL import Image

import editdistance
import mxnet as mx
import numpy as np

from callbacks.save_bboxes import BBOXPlotter
from networks.svhn import SVHNMultiLineResNetNetwork
from utils.datatypes import Size


Batch = namedtuple('Batch', ['data', 'label'])


def get_model(args, data_shape, output_size):
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.model_epoch)

    net, loc, transformed_output, size_params = SVHNMultiLineResNetNetwork.get_network(
        data_shape,
        output_size,
        23,
        1,
        1,
    )

    output = mx.symbol.Group([loc, transformed_output, net])

    context = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mx.mod.Module(output, context=context)

    model.bind(
        data_shapes=[
            ('data', data_shape),
            ('softmax_label', (1, 23)),
            ('l0_forward_init_h_state', (1, 256)),
            ('l0_forward_init_c_state_cell', (1, 256))
        ],
        for_training=False,
        grad_req='null'
    )

    arg_params['l0_forward_init_h_state'] = mx.nd.zeros((1, 256))
    arg_params['l0_forward_init_c_state_cell'] = mx.nd.zeros((1, 256))

    model.set_params(arg_params, aux_params)
    return model


def plot_bboxes(bbox_plotter, model, input_image, iteration):
    outputs = model.get_outputs()

    transform_params, interpolated_areas = bbox_plotter.get_area_data(
        1,
        outputs[0],
        outputs[1],
    )

    bbox_plotter.save_extracted_regions(
        input_image,
        interpolated_areas.asnumpy(),
        transform_params.asnumpy(),
        iteration,
        ''
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that evaluates a trained model on given eval data')
    parser.add_argument('model_prefix', help='path to saved model')
    parser.add_argument('model_epoch', type=int, help='epoch to load')
    parser.add_argument('eval_gt', help='path to evaluation csv file containing image and gt labels')
    parser.add_argument('char_map', help='path to char map for mapping classes to characters')
    parser.add_argument('--gpus', help='gpus to use for evaluation e.g. 0,1,2,3 if you want to use 4 gpus')
    parser.add_argument('--delimiter', default=';', help='delimiter used in gt file')
    parser.add_argument('--input-width', default=200, type=int, help='input width for network')
    parser.add_argument('--input-height', default=200, type=int, help='input height for network')
    parser.add_argument('--blank_symbol', type=int, default=9250, help='utf-8 code for no char class')
    parser.add_argument('--plot', action='store_true', default=False, help='plot bbox predictions and extracted regions')

    args = parser.parse_args()

    with open(args.char_map) as char_map_file:
        char_map = json.load(char_map_file)

    reverse_char_map = {v: k for k, v in char_map.items()}

    data_shape = (1, 1, args.input_height, args.input_width)
    input_size = Size(width=data_shape[-1], height=data_shape[-2])
    output_size = Size(width=50, height=50)

    model = get_model(args, data_shape, output_size)

    bbox_plotter = None
    if args.plot:
        bbox_plotter = BBOXPlotter(
            input_size,
            output_size,
            os.path.dirname(os.path.dirname(args.model_prefix)),
            bbox_dir='eval_bboxes',
        )

    num_correct = 0
    num_overall = 0
    with open(args.eval_gt) as eval_gt:
        reader = csv.reader(eval_gt, delimiter=args.delimiter)
        for idx, line in enumerate(reader):
            file_name = line[0]
            label = line[1].strip()
            gt_word = label.lower()
            label = [reverse_char_map.get(ord(char.lower()), reverse_char_map[args.blank_symbol]) for char in gt_word]
            label += [reverse_char_map[args.blank_symbol]] * (23 - len(label))

            the_image = Image.open(file_name)
            the_image = the_image.convert('L')
            the_image = the_image.resize((input_size.width, input_size.height), Image.ANTIALIAS)

            image = np.asarray(the_image, dtype=np.float32)[np.newaxis, np.newaxis, ...]
            image /= 255

            input_batch = Batch(data=[mx.nd.array(image)], label=[mx.nd.array(label)])
            model.forward(input_batch, is_train=False)

            if args.plot:
                plot_bboxes(bbox_plotter, model, image, idx)

            predictions = model.get_outputs()[2].asnumpy()
            predicted_classes = np.argmax(predictions, axis=1)

            # cut all word end predictions
            try:
                first_no_char_prediction = list(predicted_classes).index(int(reverse_char_map[args.blank_symbol]))
            except ValueError:
                first_no_char_prediction = len(predicted_classes) - 1
            predicted_classes = predicted_classes[:first_no_char_prediction]

            predicted_word = ''.join([chr(char_map[str(p)]) for p in predicted_classes])

            distance = editdistance.eval(gt_word, predicted_word)
            print("{} - {}\t\t{}: {}".format(idx, gt_word, predicted_word, distance))
            results = [prediction == label for prediction, label in zip(predicted_word, gt_word)]
            if all(results):
                num_correct += 1
            num_overall += 1

    print("Accuracy: {}".format(num_correct / num_overall))
