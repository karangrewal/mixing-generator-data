""" DCGAN for LSUN dataset """

from lasagne.layers import batch_norm, Conv2DLayer, DenseLayer, InputLayer, ReshapeLayer, TransposedConv2DLayer
from lasagne.nonlinearities import LeakyRectify, tanh

DIM_X = 64
DIM_Y = 64
DIM_C = 3

def generator(input_var=None, dim_z=128):
	layer = InputLayer(shape=(None, dim_z), input_var=input_var)
	layer = batch_norm(DenseLayer(layer, 1024 * 4 * 4))
	layer = ReshapeLayer(layer, ([0], 1024, 4, 4))
	layer = batch_norm(TransposedConv2DLayer(layer, 512, 4, stride=2, crop=1))
	layer = batch_norm(TransposedConv2DLayer(layer, 256, 4, stride=2, crop=1))
	layer = batch_norm(TransposedConv2DLayer(layer, 128, 4, stride=2, crop=1))
	layer = TransposedConv2DLayer(layer, DIM_C, 4, stride=2, crop=1)
	return layer

def discriminator(input_var=None, use_batch_norm=True, leak=0.02):
	if not use_batch_norm:
		bn = lambda x: x
	else:
		bn = batch_norm
	lrelu = LeakyRectify(leak)
	layer = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=input_var)
	layer = bn(Conv2DLayer(layer, 64, 4, stride=2, pad=2, nonlinearity=lrelu))
	layer = bn(Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu))
	layer = DenseLayer(layer, 1024, nonlinearity=lrelu)
	layer = DenseLayer(layer, 1, nonlinearity=tanh)
	return layer

