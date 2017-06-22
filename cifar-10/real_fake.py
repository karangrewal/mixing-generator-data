"""
Real and Fake discriminators for detecting fraud samples from gaussians with
unit variance centered at 1 and 0 respectively.
"""

from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import linear

N_CHANNELS = 3
N_ROWS = 32
N_COLS = 32
DIM_H = 128
DIM_Z = 100

def real_fake_discriminator(input_var=None):
	layer = InputLayer(shape=(None,1), input_var=input_var)
	layer = DenseLayer(layer, 28)
	layer = DenseLayer(layer, 1, nonlinearity=linear)
	return layer
