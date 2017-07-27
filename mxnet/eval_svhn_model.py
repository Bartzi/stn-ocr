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
from metrics.ctc_metrics import strip_prediction
from networks.svhn import SVHNMultiLineCTCNetwork
from utils.datatypes import Size

from operations.disable_shearing import *


Batch = namedtuple('Batch', ['data', 'label'])


def get_model(args, data_shape, output_size):
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.model_epoch)

    net, loc, transformed_output, size_params = SVHNMultiLineCTCNetwork.get_network(
        data_shape,
        output_size,
        11,
        2,
        3,
    )

    output = mx.symbol.Group([loc, transformed_output, net])

    context = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mx.mod.Module(output, context=context)

    model.bind(
        data_shapes=[
            ('data', data_shape),
            ('softmax_label', (1, 3)),
            ('l0_forward_init_h_state', (1, 1, 256)),
            ('l0_forward_init_c_state_cell', (1, 1, 256)),
            ('l1_forward_init_h_state', (1, 1, 256)),
            ('l1_forward_init_c_state_cell', (1, 1, 256)),
        ],
        for_training=False,
        grad_req='null'
    )

    arg_params['l0_forward_init_h_state'] = mx.nd.zeros((1, 1, 256))
    arg_params['l0_forward_init_c_state_cell'] = mx.nd.zeros((1, 1, 256))
    arg_params['l1_forward_init_h_state'] = mx.nd.zeros((1, 1, 256))
    arg_params['l1_forward_init_c_state_cell'] = mx.nd.zeros((1, 1, 256))

    model.set_params(arg_params, aux_params)
    return model


def plot_bboxes(bbox_plotter, model, input_image, iteration):
    outputs = model.get_outputs()

    transform_params, interpolated_areas = bbox_plotter.get_area_data(
        1,
        outputs[0],
        outputs[1],
    )

    labels = strip_prediction(np.argmax(outputs[2].asnumpy(), axis=1), args.blank_label)
    labels = ''.join([chr(char_map[str(x)]) for x in labels])

    bbox_plotter.save_extracted_regions(
        input_image,
        interpolated_areas.asnumpy(),
        transform_params.asnumpy(),
        iteration,
        labels,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that evaluates a trained model on given eval data')
    parser.add_argument('model_prefix', help='path to saved model')
    parser.add_argument('model_epoch', type=int, help='epoch to load')
    parser.add_argument('eval_gt', help='path to evaluation csv file containing image and gt labels')
    parser.add_argument('char_map', help='path to char map for mapping classes to characters')
    parser.add_argument('--gpus', help='gpus to use for evaluation e.g. 0,1,2,3 if you want to use 4 gpus')
    parser.add_argument('--delimiter', default=';', help='delimiter used in gt file')
    parser.add_argument('--input-width', default=64, type=int, help='input width for network')
    parser.add_argument('--input-height', default=64, type=int, help='input height for network')
    parser.add_argument('--blank_symbol', type=int, default=9250, help='utf-8 code for no char class')
    parser.add_argument('--blank-label', type=int, default=0, help='label for blank symbol [default: 0]')
    parser.add_argument('--plot', action='store_true', default=False, help='plot bbox predictions and extracted regions')
    parser.add_argument('--save-predictions', help='path to csv file where predictions shall be saved for later use')

    args = parser.parse_args()

    with open(args.char_map) as char_map_file:
        char_map = json.load(char_map_file)

    reverse_char_map = {v: k for k, v in char_map.items()}

    data_shape = (1, 1, args.input_height, args.input_width)
    input_size = Size(width=data_shape[-1], height=data_shape[-2])
    output_size = Size(width=40, height=40)

    model = get_model(args, data_shape, output_size)

    bbox_plotter = None
    if args.plot:
        bbox_plotter = BBOXPlotter(
            input_size,
            output_size,
            os.path.dirname(os.path.dirname(args.model_prefix)),
            bbox_dir='eval_bboxes',
            show_labels=True,
            label_map=args.char_map,
            blank_label=args.blank_label,
        )

    num_correct = 0
    num_overall = 0
    model_predictions = []
    with open(args.eval_gt) as eval_gt:
        reader = csv.reader(eval_gt, delimiter='\t')
        for idx, line in enumerate(reader):
            if line[0] == 'filename' and 'text' in line[1]:
                continue

            file_name = line[0]
            label = line[1:]
            
            gt_word = ''.join(chr(char_map[str(l)]) for l in label)
            gt_word = gt_word.strip(chr(char_map[str(args.blank_label)]))

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
            predicted_classes = strip_prediction(predicted_classes, int(reverse_char_map[args.blank_symbol]))
            predicted_word = ''.join([chr(char_map[str(p)]) for p in predicted_classes]).replace(' ', '')
            model_predictions.append((file_name, predicted_word))

            distance = editdistance.eval(gt_word, predicted_word)
            print("{} - {}\t\t{}: {}".format(idx, gt_word, predicted_word, distance))
            results = [prediction == label for prediction, label in zip(predicted_word, gt_word)]
            if all(results):
                num_correct += 1
            num_overall += 1

    print("Accuracy: {}".format(num_correct / num_overall))

    if args.save_predictions:
        with open(args.save_predictions, 'w') as pred_file:
            writer = csv.writer(pred_file, delimiter=';')
            writer.writerows(model_predictions)
