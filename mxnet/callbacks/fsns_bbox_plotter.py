from collections import Counter

import numpy as np

from callbacks.save_bboxes import BBOXPlotter
from utils.datatypes import Size


class FSNSBBOXPlotter(BBOXPlotter):

    def draw_bbox(self, bboxes, output_size, target_size, draw, colour):
        bboxes = np.squeeze(bboxes)
        bboxes = (bboxes + 1) / 2

        single_image_target_size = Size(width=target_size.width // 4, height=target_size.height)

        bboxes[:, 0] *= single_image_target_size.width
        bboxes[:, 1] *= single_image_target_size.height

        for idx, bbox in enumerate(np.split(bboxes, len(bboxes), axis=0)):
            bbox = np.squeeze(bbox, axis=0)

            x = np.clip(bbox[0].reshape(output_size.height, output_size.width), 0, single_image_target_size.width)
            y = np.clip(bbox[1].reshape(output_size.height, output_size.width), 0, single_image_target_size.height)

            x += idx * single_image_target_size.width

            top_left = (x[0, 0], y[0, 0])
            top_right = (x[0, -1], y[0, -1])
            bottom_left = (x[-1, 0], y[-1, 0])
            bottom_right = (x[-1, -1], y[-1, -1])

            draw.polygon(
                [top_left, top_right, bottom_right, bottom_left],
                outline=colour,
            )

    def create_image_from_array(self, array, resize=True):
        array = np.transpose(array, (0, 2, 3, 1, 4))
        shape = array.shape
        array = np.reshape(array, shape[:3] + (-1, ))
        return super(FSNSBBOXPlotter, self).create_image_from_array(array, resize=resize)

    @staticmethod
    def majority_vote(array):
        array = array.reshape(4, -1)
        votes = np.zeros((array.shape[1]), dtype=np.int32)

        for idx, result in enumerate(np.split(array, array.shape[1], axis=1)):
            result = np.squeeze(result, axis=1)
            votes[idx] = Counter(result).most_common(1)[0][0]

        return votes

    def save_extracted_regions(self, input, regions, transform_params, iteration, labels, gt_bboxes=None, extra_transform_params=None, extra_interpolated_areas=None):
        shape = regions.shape
        regions = np.reshape(regions, (shape[0] // 4, 4,) + shape[1:])
        shape = transform_params.shape
        transform_params = np.reshape(transform_params, (shape[0] // 4, 4) + shape[1:])
        super(FSNSBBOXPlotter, self).save_extracted_regions(
            input,
            regions,
            transform_params,
            iteration,
            labels,
            gt_bboxes=gt_bboxes,
            extra_transform_params=extra_transform_params,
            extra_interpolated_areas=extra_interpolated_areas,
        )
