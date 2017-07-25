import json
import math
import socket

import base64
import os
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageOps

import mxnet as mx
import numpy as np

from PIL import Image, ImageDraw

from metrics.ctc_metrics import strip_prediction
from utils.bbox_utils import get_sampling_grid
from utils.datatypes import Size

COLOR_MAP = [
    0xFFB300,  # Vivid Yellow
    0x803E75,  # Strong Purple
    0xFF6800,  # Vivid Orange
    0xA6BDD7,  # Very Light Blue
    0xC10020,  # Vivid Red
    0xCEA262,  # Grayish Yellow
    0x817066,  # Medium Gray

    # The following don't work well for people with defective color vision
    0x007D34,  # Vivid Green
    0xF6768E,  # Strong Purplish Pink
    0x00538A,  # Strong Blue
    0xFF7A5C,  # Strong Yellowish Pink
    0x53377A,  # Strong Violet
    0xFF8E00,  # Vivid Orange Yellow
    0xB32851,  # Strong Purplish Red
    0xF4C800,  # Vivid Greenish Yellow
    0x7F180D,  # Strong Reddish Brown
    0x93AA00,  # Vivid Yellowish Green
    0x593315,  # Deep Yellowish Brown
    0xF13A13,  # Vivid Reddish Orange
    0x232C16,  # Dark Olive Green

    # extend colour map
    0xFFB300,  # Vivid Yellow
    0x803E75,  # Strong Purple
    0xFF6800,  # Vivid Orange
    0xA6BDD7,  # Very Light Blue
    0xC10020,  # Vivid Red
    0xCEA262,  # Grayish Yellow
    0x817066,  # Medium Gray
]


class BBOXPlotter:

    def __init__(self, image_size, output_size, base_dir, **kwargs):
        self.image_size = image_size
        self.output_size = output_size
        self.base_dir = base_dir
        self.save_attention = kwargs.pop('save_attention', False)
        self.plot_extra_loc = kwargs.pop('plot_extra_loc', False)
        self.plot_individual_regions = kwargs.pop('plot_individual_regions', True)

        self.show_labels = kwargs.pop('show_labels', False)
        self.save_labels = kwargs.pop('save_labels', False)
        if self.show_labels:
            self.label_majority_vote = kwargs.pop('do_label_majority_vote', False)
            label_map = kwargs.pop('label_map')
            with open(label_map) as the_label_map:
                self.label_map = json.load(the_label_map)
                self.font = ImageFont.truetype('utils/DejaVuSans.ttf', 20)
                self.blank_label = kwargs.pop('blank_label')

        self.send_bboxes = kwargs.pop('send_bboxes', False)
        if self.send_bboxes:
            socket.setdefaulttimeout(2)
            self.upstream_ip = kwargs.pop('upstream_ip', None)
            self.upstream_port = kwargs.pop('upstream_port', 1337)

        self.bbox_dir = os.path.join(base_dir, kwargs.pop('bbox_dir', 'bboxes'))
        self.create_empty_dir(self.bbox_dir)
        if self.save_attention:
            self.attention_dir = os.path.join(base_dir, 'attention')
            self.create_empty_dir(self.attention_dir)

    @staticmethod
    def create_empty_dir(dirname):
        os.makedirs(dirname, exist_ok=True)
        for file_name in os.listdir(dirname):
            os.remove(os.path.join(dirname, file_name))

    @staticmethod
    def normalize_image(image):
        min_val = image.min()
        max_val = image.max()
        if min_val != max_val:
            image -= min_val
            image *= 1.0 / (max_val - min_val)
        else:
            image[:] = 1.0
        return image

    def draw_bbox(self, bbox, output_size, target_size, draw, colour):
        bbox = np.squeeze(bbox)
        bbox = (bbox + 1) / 2

        bbox[0] *= target_size.width
        bbox[1] *= target_size.height
        x = np.clip(bbox[0].reshape(output_size.height, output_size.width), 0, target_size.width)
        y = np.clip(bbox[1].reshape(output_size.height, output_size.width), 0, target_size.height)

        top_left = (x[0, 0], y[0, 0])
        top_right = (x[0, -1], y[0, -1])
        bottom_left = (x[-1, 0], y[-1, 0])
        bottom_right = (x[-1, -1], y[-1, -1])

        draw.polygon(
            [top_left, top_right, bottom_right, bottom_left],
            outline=colour,
        )

    def create_image_from_array(self, array, resize=True):
        array = np.squeeze(array)
        shape = array.shape
        is_rgb = len(shape) == 3 and shape[0] == 3
        if len(shape) > 2 and not is_rgb:
            best_region_index = np.argmax(np.mean(np.reshape(array, (len(array), -1)), axis=1))
            array = array[best_region_index]
        array = array * 255 if array.max() <= 1 else array
        if is_rgb:
            array = np.transpose(array, (1, 2, 0))

        image = Image.fromarray(array.astype(np.uint8), mode='RGB' if is_rgb else 'L')

        if not is_rgb:
            image = image.convert('RGB')

        if resize:
            image = image.resize(self.image_size)
        return image

    def send_image(self, data):
        height = data.height
        width = data.width
        data = np.asarray(data, dtype=np.uint8).tobytes()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.upstream_ip, self.upstream_port))
            except Exception as e:
                print(e)
                print("could not connect to display server, disabling image rendering")
                self.send_bboxes = False
                return
            data = {
                'width': width,
                'height': height,
                'image': base64.b64encode(data).decode('utf-8'),
            }
            sock.send(bytes(json.dumps(data), 'utf-8'))

    def save_extracted_regions(self, input, regions, transform_params, iteration, labels, gt_bboxes=None, extra_transform_params=None, extra_interpolated_areas=None):
        sampling_grids = get_sampling_grid(transform_params, self.output_size)

        if extra_transform_params is not None:
            extra_size = Size(width=self.output_size.width // 2, height=self.output_size.height // 2)
            extra_sampling_grids = get_sampling_grid(
                extra_transform_params,
                extra_size,
            )
        else:
            extra_sampling_grids = np.zeros_like(sampling_grids)
            extra_interpolated_areas = np.zeros_like(sampling_grids)

        if self.plot_individual_regions or iteration % 1000 == 0:
            dest_image_size = ((len(regions) + 1) * self.image_size.width, self.image_size.height if not self.plot_extra_loc else 2 * self.image_size.height)
        else:
            dest_image_size = (self.image_size.width, self.image_size.height)
        dest_image = Image.new('RGB', dest_image_size)

        image = self.create_image_from_array(input, resize=False)
        draw = ImageDraw.Draw(image)

        data_iter = zip(
            np.split(regions, len(regions)),
            np.split(sampling_grids, len(sampling_grids)),
            np.split(extra_sampling_grids, len(extra_sampling_grids)),
            np.split(extra_interpolated_areas, len(extra_interpolated_areas)),
            COLOR_MAP,
        )

        for idx, (region, bbox, extra_bbox, extra_region, colour) in enumerate(data_iter, start=1):
            # show each individual region if the user wants it or after every 1000 iterations for debugging purposes
            if self.plot_individual_regions or iteration % 1000 == 0:
                region_image = self.create_image_from_array(region)

                if self.plot_extra_loc:
                    region_draw = ImageDraw.Draw(region_image)
                    self.draw_bbox(extra_bbox, extra_size, self.image_size, region_draw, colour)
                    extra_region_image = self.create_image_from_array(extra_region)
                    dest_image.paste(extra_region_image, (idx * self.image_size.width, self.image_size.height))

                dest_image.paste(region_image, (idx * self.image_size.width, 0))

            # draw bbox
            self.draw_bbox(bbox, self.output_size, self.image_size, draw, colour)

        if gt_bboxes is not None:
            for gt_bbox, colour in zip(np.split(gt_bboxes, len(gt_bboxes)), reversed(COLOR_MAP)):
                gt_bbox = np.squeeze(gt_bbox, axis=0)
                top_left = (gt_bbox[0], gt_bbox[2])
                top_right = (gt_bbox[0] + gt_bbox[1], gt_bbox[2])
                bottom_left = (gt_bbox[0], gt_bbox[2] + gt_bbox[3])
                bottom_right = (gt_bbox[0] + gt_bbox[1], gt_bbox[2] + gt_bbox[3])

                draw.polygon(
                    [top_left, top_right, bottom_right, bottom_left],
                    outline=colour,
                )

        if self.show_labels:
            # only keep ascii characters
            # labels = ''.join(filter(lambda x: len(x) == len(x.encode()), labels))
            text_width, text_height = draw.textsize(labels, font=self.font)
            draw = ImageDraw.Draw(dest_image)
            draw.text((dest_image.width - text_width - 1, 0), labels, fill='green', font=self.font)

        dest_image.paste(image, (0, 0))
        if self.save_labels:
            file_name = "{}_{}.png".format(os.path.join(self.bbox_dir, str(iteration)), labels)
        else:
            file_name = "{}.png".format(os.path.join(self.bbox_dir, str(iteration)))
        dest_image.save(file_name)

        if self.send_bboxes:
            self.send_image(dest_image)

    @staticmethod
    def copy_params(params, original_executor, attr_name='param'):
        for param_name in params.keys():
            if param_name == "data" or param_name == "softmax_label":
                continue
            if param_name.endswith('state') or param_name.endswith('state_cell'):
                continue
            try:
                param_index = getattr(original_executor, "{}_names".format(attr_name)).index(param_name)
                param_data = getattr(original_executor, "{}_arrays".format(attr_name))[param_index]
                params[param_name][:] = param_data[0]
            except ValueError:
                continue
        return params

    def get_area_data(self, batch_size, transform_params, interpolated_areas):
        size, num_params = transform_params.shape
        transform_params = transform_params.reshape((size // batch_size, batch_size, num_params))
        transform_params = mx.nd.transpose(transform_params, axes=(1, 0, 2))[0]

        size, num_channels, height, width = interpolated_areas.shape
        interpolated_areas = interpolated_areas.reshape(
            (size // batch_size, batch_size, num_channels, height, width))
        interpolated_areas = mx.nd.transpose(interpolated_areas, axes=(1, 0, 2, 3, 4))[0]
        return transform_params, interpolated_areas

    @staticmethod
    def majority_vote(array):
        raise NotImplementedError("This class can not handle data that might need majority voting on labels")

    def get_callback(self, stn_output, data, label, num_data=1, batch_num=None, show_gt_bboxes=False):
        """
            function that creates an image showing the current state of the network when it comes to predicting
            the locations of the ROIs, found by the spatial transformer. In order to do this this callback performs a forward
            pass through the network at its current state. This callback saves one image for each call.
        :param stn_output: symbol that is the grouped output of the localization net and the spatial transformer
        :param data: np array of input data in the format as it is expected by the network (i.e. shape=(batch_size, channels, height, width))
        :param label:
        :param dest_dir: destination directory where the resulting bbox plots shall be saved.
        :return: a callback function
        """
        def plot_bboxes(execution_params):
            data_iter = execution_params.locals['train_data']
            iters_per_epoch = num_data // data_iter.batch_size

            batch_size = data_iter.batch_size if batch_num is None else batch_num

            input_data_shapes = {description.name: (batch_size, ) + description.shape[1:] for description in data_iter.provide_data}

            for label_data in data_iter.provide_label:
                input_data_shapes[label_data.name] = (batch_size, ) + label_data.shape[1:]

            executor = stn_output.simple_bind(execution_params.locals['ctx'][0], grad_req='null', **input_data_shapes)

            # set weights of executor
            original_executor = execution_params.locals['executor_manager']
            params = executor.arg_dict
            self.copy_params(params, original_executor, attr_name='param')

            aux_params = executor.aux_dict
            self.copy_params(aux_params, original_executor, attr_name='aux')

            params['data'][:] = mx.nd.array(np.tile(data, (batch_size, 1, 1, 1)))
            params['softmax_label'] = mx.nd.array(label)

            executor.forward(is_train=False)

            transform_params, interpolated_areas = self.get_area_data(
                batch_size,
                executor.outputs[0],
                executor.outputs[1],
            )

            if self.show_labels:
                if self.label_majority_vote:
                    labels = self.majority_vote(np.argmax(executor.outputs[2].asnumpy(), axis=1))
                    labels = strip_prediction(labels, self.blank_label)
                else:
                    labels = strip_prediction(np.argmax(executor.outputs[2].asnumpy(), axis=1), self.blank_label)
                labels = ''.join([chr(self.label_map[str(x)]) for x in labels])
            else:
                labels = ''

            extra_transform_params = None
            extra_interpolated_areas = None
            if self.plot_extra_loc:
                extra_transform_params = executor.outputs[3]
                size, num_params = extra_transform_params.shape
                extra_transform_params = extra_transform_params.reshape((size // batch_size, batch_size, num_params))
                extra_transform_params = mx.nd.transpose(extra_transform_params, axes=(1, 0, 2))[0]

                extra_interpolated_areas = executor.outputs[4]
                size, num_channels, height, width = extra_interpolated_areas.shape
                extra_interpolated_areas = extra_interpolated_areas.reshape((size // batch_size, batch_size, num_channels, height, width))
                extra_interpolated_areas = mx.nd.transpose(extra_interpolated_areas, axes=(1, 0, 2, 3, 4))[0]

            gt_bboxes = None
            if show_gt_bboxes:
                num_timesteps = size // batch_size
                _, gt_bboxes = np.split(label, [-(num_timesteps * 4)])
                gt_bboxes = gt_bboxes.reshape(num_timesteps, 4)

            iteration = execution_params.epoch * iters_per_epoch + execution_params.nbatch
            self.save_extracted_regions(
                data,
                interpolated_areas.asnumpy(),
                transform_params.asnumpy(),
                iteration,
                labels,
                gt_bboxes=gt_bboxes,
                extra_transform_params=extra_transform_params.asnumpy() if extra_transform_params is not None else None,
                extra_interpolated_areas=extra_interpolated_areas.asnumpy() if extra_interpolated_areas is not None else None,
            )

        return plot_bboxes
