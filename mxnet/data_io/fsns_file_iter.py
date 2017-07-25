import mxnet as mx
import numpy as np

from data_io.file_iter import FileBasedIter


class FSNSFileIter(FileBasedIter):

    @property
    def provide_data(self):
        return [
            mx.io.DataDesc("data", (self.batch_size, 4) + self.image_shape[0:2] + (self.image_shape[-1] // 4,), self.image_dtype)
        ]

    def next(self):
        if self.overflowed:
            return StopIteration

        images = []
        labels = []
        while len(images) < self.batch_size and len(labels) < self.batch_size:
            success, image, label = self.done_queue.get()
            if not success:
                self.task_queue.put(self.lines[self.indices[self.cursor]])
                self.cursor += 1
                if self.cursor >= self.num_data:
                    return StopIteration
                continue

            image = image.reshape(1, -1, self.resize_to.height, 4, self.resize_to.width // 4)
            image = image.transpose(0, 3, 1, 2, 4)

            images.append(image)
            labels.append(label)

        data = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)

        self.invoke_prefetch()
        return mx.io.DataBatch(data=[mx.nd.array(data)], label=[mx.nd.array(labels)], pad=[self.get_pad()], index=None)
