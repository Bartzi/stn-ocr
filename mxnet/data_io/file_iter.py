import csv
from queue import Empty

import os
from multiprocessing import Queue
from multiprocessing import Process

import mxnet as mx
import numpy as np

from PIL import Image


class FileBasedIter(mx.io.DataIter):

    def __init__(self, dataset_file, batch_size, expected_label_length, label_name='softmax_label', resize_to=None, base_dir=None, delimiter=',', num_workers=4, image_mode='L'):
        super(FileBasedIter, self).__init__()
        self.base_dir = os.path.dirname(dataset_file) if base_dir is None else base_dir
        self.resize_to = resize_to
        self.num_workers = num_workers
        self.lines = []

        if image_mode not in ["L", "RGB"]:
            raise ValueError("Please use only 'L' (black and white) or 'RGB' as color mode specifiers")

        with open(dataset_file, 'r') as f:
            self.lines = [l for l in f]

        self.indices = np.random.permutation(len(self.lines))

        self.batch_size = batch_size
        self.overflowed = False
        self.label_name = label_name

        image = _load_image(self.lines[0].split(delimiter)[0], self.base_dir, self.resize_to, image_mode=image_mode)
        self.image_shape = image.shape
        self.image_dtype = image.dtype
        label = np.array(self.lines[0].split(delimiter)[1:], dtype=np.int32)
        self.label_shape = label.shape
        self.label_dtype = image.dtype

        self.hard_reset()
        self.workers = []
        for _ in range(self.num_workers):
            worker = Process(
                target=_load_worker,
                args=(
                    self.task_queue,
                    self.done_queue,
                    self.base_dir,
                    self.resize_to,
                    delimiter,
                    expected_label_length,
                    image_mode,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

    def shutdown(self):
        try:
            while True:
                self.done_queue.get_nowait()
        except Empty:
            pass

        for _ in range(2 * self.num_workers):
            self.task_queue.put(None)

    @property
    def num_data(self):
        return len(self.lines)

    @property
    def provide_data(self):
        return [
            mx.io.DataDesc("data", (self.batch_size, ) + self.image_shape, self.image_dtype)
        ]

    @property
    def provide_label(self):
        return [
            mx.io.DataDesc(self.label_name, (self.batch_size, ) + self.label_shape, self.label_dtype)
        ]

    def hard_reset(self):
        self.cursor = 0
        self.task_queue = Queue(self.batch_size * 10)
        self.done_queue = Queue(self.batch_size * 3)
        self.overflowed = False
        self.invoke_prefetch()

    def reset(self):
        self.cursor = (self.cursor % self.num_data) % self.batch_size
        self.overflowed = False

    def iter_next(self):
        return self.cursor < self.num_data

    def invoke_prefetch(self):
        for _ in range(self.batch_size):
            self.task_queue.put(self.lines[self.indices[self.cursor]])
            self.cursor += 1
            if self.cursor >= self.num_data:
                self.cursor = 0
                self.overflowed = True

    def next(self):
        if self.overflowed:
            raise StopIteration

        images = []
        labels = []
        while len(images) < self.batch_size and len(labels) < self.batch_size:
            success, image, label = self.done_queue.get()
            if not success:
                self.task_queue.put(self.lines[self.indices[self.cursor]])
                self.cursor += 1
                if self.cursor >= self.num_data:
                    raise StopIteration
                continue

            images.append(image)
            labels.append(label)

        data = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        self.invoke_prefetch()
        return mx.io.DataBatch(data=[mx.nd.array(data)], label=[mx.nd.array(labels)], pad=[self.get_pad()], index=None)

    def getdata(self):
        raise NotImplementedError("Please use next nethod!")

    def getlabel(self):
        raise NotImplementedError("Please use next nethod!")

    def get_pad(self):
        return 0


def _load_image(file_name, base_dir, target_size=None, image_mode='L'):
    # the_image = Image.open(os.path.join(base_dir, file_name))
    with Image.open(os.path.join(base_dir, file_name)) as the_image:
        the_image = the_image.convert(image_mode)
        if target_size is not None:
            the_image = the_image.resize((target_size.width, target_size.height), Image.ANTIALIAS)

        image = np.asarray(the_image, dtype=np.float32)
        image /= 255
        if image_mode != "L":
            return image.transpose(2, 0, 1)
        else:
            return image[np.newaxis, ...]


def _load_worker(input_queue, output_queue, base_dir, target_size, delimiter, expected_label_length, image_mode):
    while True:
        line = input_queue.get()
        if line is None:
            break
        line = line.split(delimiter)
        file_name = line[0]
        labels = np.array(line[1:], dtype=np.int32)
        image = None

        try:
            success = True
            image = _load_image(file_name, base_dir, target_size, image_mode=image_mode)[np.newaxis, ...]
            if len(labels) != expected_label_length:
                success = False
            else:
                labels = labels[np.newaxis, ...]
        except Exception:
            success = False
        output_queue.put((success, image, labels))