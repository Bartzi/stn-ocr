import argparse
import csv
import json
import os
from collections import namedtuple

from PIL import Image

import editdistance
import mxnet as mx
import numpy as np

from callbacks.fsns_bbox_plotter import FSNSBBOXPlotter
from callbacks.save_bboxes import BBOXPlotter
from metrics.ctc_metrics import strip_prediction
from networks.fsns import FSNSNetwork
from networks.text_rec import SVHNMultiLineResNetNetwork, SVHNMultiLineCTCNetwork
from utils.datatypes import Size

from operations.disable_shearing import *


Batch = namedtuple('Batch', ['data', 'label'])


def get_model(args, data_shape, output_size):
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.model_epoch)

    net, loc, transformed_output, size_params = FSNSNetwork.get_network(
        data_shape,
        output_size,
        3,
        1,
        10,
        blstm=True,
    )

    output = mx.symbol.Group([loc, transformed_output, net])

    context = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mx.mod.Module(output, context=context)

    model.bind(
        data_shapes=[
            ('data', data_shape),
            ('softmax_label', (1, 30)),
            ('l0_forward_init_h_state', (1, 4, 256)),
            ('l0_forward_init_c_state_cell', (1, 4, 256)),
            ('l0_backward_init_h_state', (1, 4, 256)),
            ('l0_backward_init_c_state_cell', (1, 4, 256))
        ],
        for_training=False,
        grad_req='null'
    )

    arg_params['l0_forward_init_h_state'] = mx.nd.zeros((1, 4, 256))
    arg_params['l0_forward_init_c_state_cell'] = mx.nd.zeros((1, 4, 256))
    arg_params['l0_backward_init_h_state'] = mx.nd.zeros((1, 4, 256))
    arg_params['l0_backward_init_c_state_cell'] = mx.nd.zeros((1, 4, 256))

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
    parser.add_argument('--blank-label', type=int, default=0, help='label for blank symbol [default: 0]')
    parser.add_argument('--plot', action='store_true', default=False, help='plot bbox predictions and extracted regions')
    parser.add_argument('--save-predictions', help='path to csv file where predictions shall be saved for later use')

    args = parser.parse_args()

    with open(args.char_map) as char_map_file:
        char_map = json.load(char_map_file)

    reverse_char_map = {v: k for k, v in char_map.items()}

    args.input_height = 150
    args.input_width = 600
    data_shape = (1, 4, 3, args.input_height, args.input_width // 4)
    input_size = Size(width=data_shape[-1], height=data_shape[-2])
    output_size = Size(width=75, height=50)

    model = get_model(args, data_shape, output_size)

    bbox_plotter = None
    if args.plot:
        bbox_plotter = FSNSBBOXPlotter(
            input_size,
            output_size,
            os.path.dirname(os.path.dirname(args.model_prefix)),
            bbox_dir='eval_bboxes',
            show_labels=True,
            label_map=args.char_map,
            blank_label=args.blank_label,
            plot_extra_loc=False,
            plot_individual_regions=False,
            do_label_majority_vote=False,
            save_attention=False,
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
            labels = line[1:]
            label_1 = labels[:10]
            label_2 = labels[10:20]
            label_3 = labels[20:30]

            the_image = Image.open(file_name)

            # reinterpret image data from one large image into 4 individual images
            image = np.asarray(the_image, dtype=np.float32)[np.newaxis, ...]
            image = image.transpose(0, 3, 1, 2)
            image /= 255
            image = image.reshape(1, -1, args.input_height, 4, args.input_width // 4)
            image = image.transpose(0, 3, 1, 2, 4)

            input_batch = Batch(data=[mx.nd.array(image)], label=[mx.nd.array(labels)])
            model.forward(input_batch, is_train=False)

            if args.plot:
                plot_bboxes(bbox_plotter, model, image, idx)

            # extract predictions from model
            predictions = model.get_outputs()[2].asnumpy()
            # get predicted classes
            predicted_classes = np.argmax(predictions, axis=1)
            # interpret batch or predictions as three different predictions (for each time step one prediction)
            predicted_classes = predicted_classes.reshape(3, -1)

            # strip blanks and double symbols from prediction
            predicted_classes = [strip_prediction(predicted_classes[i], args.blank_label) for i in range(len(predicted_classes))]
            # concat groundtruth
            gt_words = [l for l in [label_1, label_2, label_3]]
            # convert groundtruth from labels to chars
            gt_words = [''.join([chr(char_map[str(p)]) for p in g]) for g in gt_words]
            # strip blank labels from groundtruth
            gt_words = [g.strip(chr(char_map[str(args.blank_label)])) for g in gt_words]
        
            # convert predicted classes to characters
            predicted_words = [''.join([chr(char_map[str(g)]) for g in p]) for p in predicted_classes]
            model_predictions.append((file_name, predicted_words))

            # eval predictions
            for gt_word, predicted_word in zip(gt_words, predicted_words):
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
