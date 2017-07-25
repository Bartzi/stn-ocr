from mxnet.io import DataIter

import mxnet as mx


class DataIterDecorator(DataIter):
    def __getattribute__(self, item):
        try:
            v = object.__getattribute__(self, item)
        except AttributeError:
            v = getattr(object.__getattribute__(self, 'iter'), item)
        return v

    def __init__(self, *args, **kwargs):
        self.iter = kwargs.pop('base_iter')

        super(DataIterDecorator, self).__init__(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def getindex(self):
        return self.iter.getindex()

    def getdata(self):
        return self.iter.getdata()

    def getlabel(self):
        return self.iter.getlabel()

    @property
    def provide_data(self):
        return self.iter.provide_data

    @property
    def provide_label(self):
        return self.iter.provide_label

    def hard_reset(self):
        return self.iter.hard_reset()

    def reset(self):
        return self.iter.reset()

    def iter_next(self):
        return self.iter.iter_next()

    def next(self):
        return self.iter.next()

    def getpad(self):
        return self.iter.getpad()


class LSTMIter(DataIterDecorator):

    def getdata(self):
        return [init_state[1] for init_state in self.init_states]

    def getlabel(self):
        raise NotImplementedError("LSTM iter does not provide any label")

    @property
    def provide_data(self):
        descriptions = getattr(self.iter, 'provide_data')
        # if the data we want to add is already there (magic) then we won't do it again
        if self.init_states[0][0] in [d.name for d in descriptions]:
            return descriptions
        descriptions.extend(
            [mx.io.DataDesc(k, v.shape, v.dtype)
             for k, v in self.init_states]
        )
        return descriptions

    def next(self):
        if self.iter_next():
            iter_batch = self.iter.next()

            if iter_batch is StopIteration:
                raise StopIteration

            lstm_data = self.getdata()
            data = iter_batch.data
            if not isinstance(data, list):
                data = [data]
            data.extend(lstm_data)

            iter_batch.data = data
            return iter_batch
        else:
            raise StopIteration


class InitStateLSTMIter(LSTMIter):

    def __init__(self, *args, **kwargs):
        self.num_lstm_layers = kwargs.pop("num_lstm_layers", 1)
        self.state_size = kwargs.pop("state_size", None)
        if self.state_size is None:
            self.size_params = kwargs.pop("size_params")
        else:
            self.batch_size_multipliers = kwargs.pop('batch_size_multipliers', [])
            if len(self.batch_size_multipliers) == 0:
                self.batch_size_multipliers = [1 for _ in range(self.num_lstm_layers)]
        self.blstm = kwargs.pop("blstm", False)

        super(LSTMIter, self).__init__(*args, **kwargs)
        self.batch_size = self.iter.batch_size

        self.init_states = []
        for layer_id, batch_size_multiplier in zip(range(self.num_lstm_layers), self.batch_size_multipliers):
            if self.state_size is not None:
                lstm_init_shape = (self.iter.batch_size, batch_size_multiplier, self.state_size)
            else:
                lstm_init_shape = (self.iter.batch_size, batch_size_multiplier) + self.size_params[1:]

            self.init_states.append(('l{}_forward_init_c_state_cell'.format(layer_id), mx.nd.zeros(lstm_init_shape)))
            self.init_states.append(('l{}_forward_init_h_state'.format(layer_id), mx.nd.zeros(lstm_init_shape)))
            if self.blstm:
                backward_shape = lstm_init_shape
                self.init_states.append(
                    ('l{}_backward_init_c_state_cell'.format(layer_id), mx.nd.zeros(backward_shape)))
                self.init_states.append(
                    ('l{}_backward_init_h_state'.format(layer_id), mx.nd.zeros(backward_shape)))
