"""
Real and Fake discriminators for detecting fraud samples from gaussians with 
unit variance centered at 1 and 0 respectively.
"""

from lasagne.layers import DenseLayer, DimshuffleLayer, InputLayer
from lasagne.nonlinearities import linear

BATCH_SIZE = 64
N_CHANNELS = 1
N_ROWS = 28
N_COLS = 28
DIM_H = 64
DIM_Z = 100

def real_fake_discriminator(input_var=None):
	layer = InputLayer(shape=(BATCH_SIZE,1), input_var=input_var)
	layer = DimshuffleLayer(layer, (1, 0))
	layer = DenseLayer(layer, 28)
	layer = DenseLayer(layer, 1, nonlinearity=linear)
	return layer