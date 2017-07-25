import logging
import numpy as np
import mxnet as mx


class SPNInitializer(mx.init.Xavier):

    def __init__(self, *args, **kwargs):
        self.zoom = kwargs.pop('zoom', 0.9)
        super(SPNInitializer, self).__init__(*args, **kwargs)

    def _init_loc_bias(self, _, arr):
        shape = arr.shape
        assert (shape[0] == 6)
        # arr[:] = np.array([0.5, 0, 0, 0, 0.5, 0])
        arr[:] = np.array([self.zoom, 0, 0, 0, self.zoom, 0])


class ShapeAgnosticLoad(mx.initializer.Load):

    def load_default(self, name, arr):
        assert self.default_init is not None, \
            "Cannot Initialize %s. Not found in loaded param " % name + \
            "and no default Initializer is provided."
        self.default_init(name, arr)
        if self.verbose:
            logging.info('Initialized %s by default', name)

    def __call__(self, name, arr):
        # if name in self.param and 'rec' not in name:
        if name in self.param:
            if arr.shape != self.param[name].shape:
                self.load_default(name, arr)
                return

            arr[:] = self.param[name]
            if self.verbose:
                logging.info('Initialized %s by loading', name)
        else:
            self.load_default(name, arr)

