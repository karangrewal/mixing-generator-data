# NOTE: CIFAR-10 examples are scaled between -1 and +1.

import h5py
import numpy as np
import pickle

from fuel.datasets.cifar10 import CIFAR10
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer

import theano
floatX = theano.config.floatX

# Code taken from Devon Hjelm:
class Rescale(Transformer):
    def __init__(self, data_stream, min=-1, max=1, use_tanh=False, **kwargs):
        super(Rescale, self).__init__(data_stream=data_stream, produces_examples=False, **kwargs)
        self.min = min
        self.max = max
        self.use_tanh = use_tanh
    
    def transform_batch(self, batch):
        index = self.sources.index('features')
        x = batch[index]
        x = float(self.max - self.min) * (x / 255.) + self.min
        if self.use_tanh: x = 2. * x - 1. 
        x = x.astype(floatX)
        batch = list(batch)
        batch[index] = x
        return tuple(batch)

"""
def get_data(batch_size=64):

    # batch 1
    with open('/u/grewalka/data-cifar-10/data_batch_1') as f:
        data = pickle.load(f)['data']

    # batch 2
    with open('/u/grewalka/data-cifar-10/data_batch_2') as f:
        _data = pickle.load(f)['data']
    data = np.concatenate((data, _data), axis=0)

    # batch 3
    with open('/u/grewalka/data-cifar-10/data_batch_3') as f:
        _data = pickle.load(f)['data']
    data = np.concatenate((data, _data), axis=0)

    # batch 4
    with open('/u/grewalka/data-cifar-10/data_batch_4') as f:
        _data = pickle.load(f)['data']
    data = np.concatenate((data, _data), axis=0)

    # batch 5
    with open('/u/grewalka/data-cifar-10/data_batch_5') as f:
        _data = pickle.load(f)['data']
    data = np.concatenate((data, _data), axis=0)

    k = data.shape[0]
    data = data.reshape(k, 3, 32, 32)#.transpose(0, 2, 3, 1)
    data = np.float32(data)
    data = data / 255.
    data = data * 2.
    data = data - 1.

    k = k / batch_size
    k = k * batch_size
    k = data.shape[0] - k

    np.random.shuffle(data)
    data = data[k:]
    data = np.split(data, data.shape[0] / batch_size)
    data = np.array(data)

    return data
"""
def get_data(batch_size=64):
    dataset = CIFAR10(which_sets=['train'])
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)
    stream = DataStream(dataset, iteration_scheme=scheme)
    # data = stream.get_epoch_iterator(as_dict=True).next()['features']
    stream = Rescale(stream)
    return stream, dataset.num_examples
