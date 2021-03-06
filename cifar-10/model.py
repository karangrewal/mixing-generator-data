""" Code taken from Devon Hjelm. """

from lasagne.layers import (
    batch_norm, Conv2DLayer, DenseLayer, InputLayer, ReshapeLayer,
    TransposedConv2DLayer)
from lasagne.nonlinearities import LeakyRectify, tanh

DIM_X = 32
DIM_Y = 32
DIM_C = 3
NONLIN = tanh

def discriminator(input_var=None, use_batch_norm=True, leak=0.02):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    lrelu = LeakyRectify(leak)

    layer = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=input_var)
    layer = Conv2DLayer(layer, 128, 4, stride=2, pad=1, nonlinearity=lrelu)
    layer = bn(Conv2DLayer(layer, 256, 4, stride=2, pad=1,nonlinearity=lrelu))
    layer = bn(Conv2DLayer(layer, 512, 4, stride=2, pad=1,nonlinearity=lrelu))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    return layer

def generator(input_var=None, dim_z=128, use_batch_norm=True):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
     
    layer = bn(DenseLayer(layer, 8192))
    layer = ReshapeLayer(layer, ([0], 512, 4, 4))
    layer = bn(TransposedConv2DLayer(layer, 256, 4, stride=2, crop=1))
    layer = bn(TransposedConv2DLayer(layer, 128, 4, stride=2, crop=1))
    layer = TransposedConv2DLayer(layer, DIM_C, 4, stride=2, crop=1, nonlinearity=NONLIN)
    return layer