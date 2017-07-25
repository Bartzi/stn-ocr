import argparse
import csv
from collections import namedtuple

from PIL import Image

import mxnet as mx
import numpy as np

from detect_random_noise import get_network

Batch = namedtuple('Batch', ['data', 'label'])


def get_model(args):
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.trained_model, args.epoch)

    net = get_network()

    context = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mx.mod.Module(net, context=context)

    model.bind(
        data_shapes=[
            ('data', (1, 3, 150, 150)),
            ('softmax_label', (1, 1)),
        ],
        for_training=False,
        grad_req='null',
    )

    model.set_params(arg_params, aux_params)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that takes a list of images and decides whether the image contains random noise or not')
    parser.add_argument('image_list', help='path to image list with images that shall be examined')
    parser.add_argument('result_list', help='path to file that contains names of all images without random noise')
    parser.add_argument('trained_model', help='path to trained model that shall be used')
    parser.add_argument('epoch', type=int)
    parser.add_argument('--gpus', help='gpus to use')

    args = parser.parse_args()

    model = get_model(args)

    with open(args.image_list) as image_list, open(args.result_list, 'w') as result_list:
        reader = csv.reader(image_list, delimiter='\t')
        writer = csv.writer(result_list, delimiter='\t')

        for idx, line in enumerate(reader):
            image_path = line[0]
            the_image = Image.open(image_path)
            the_image = the_image.convert('RGB')
            the_image = the_image.resize((150, 150), Image.ANTIALIAS)

            image = np.asarray(the_image, dtype=np.float32)[np.newaxis, ...]
            image = np.transpose(image, (0, 3, 1, 2))
            image /= 255

            input_batch = Batch(data=[mx.nd.array(image)], label=[mx.nd.array([0])])
            model.forward(input_batch, is_train=False)

            prediction = model.get_outputs()[0].asnumpy()
            predicted_class = np.argmax(prediction, axis=1)

            if predicted_class == 0:
                writer.writerow(line)

            print("done with {:6} files".format(idx), end='\r')
