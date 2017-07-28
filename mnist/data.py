
import h5py
import numpy as np
import pickle

from fuel.datasets.mnist import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer

import theano
floatX = theano.config.floatX

# Code taken from Devon Hjelm:
class Rescale(Transformer):
    def __init__(self, data_stream, min=0, max=1, use_tanh=False, **kwargs):
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

def get_data(batch_size=64):
    dataset = MNIST(which_sets=['train'])
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)
    stream = DataStream(dataset, iteration_scheme=scheme)
    # data = stream.get_epoch_iterator(as_dict=True).next()['features']
    stream = Rescale(stream)
    return stream, dataset.num_examples

def get_dataset():
    dataset = MNIST(which_sets=['train'])
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=dataset.num_examples)
    stream = DataStream(dataset, iteration_scheme=scheme)
    stream = Rescale(stream)
    return stream, dataset.num_examples
