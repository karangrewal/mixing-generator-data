from lasagne.layers import batch_norm, Conv2DLayer, DenseLayer, InputLayer, MaxPool2DLayer, ReshapeLayer, TransposedConv2DLayer
from lasagne.nonlinearities import leaky_rectify, linear, rectify, sigmoid, LeakyRectify
import theano
import lasagne
import theano.tensor as T
from weight_normalization import weight_norm

N_CHANNELS = 1
N_ROWS = 28
N_COLS = 28
DIM_H = 64
DIM_Z = 100
LAYER_FN = batch_norm

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 W=None, b=None,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        if W is None:
            self.W = self.add_param(
                lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        else:
            self.W = self.add_param(
                W, (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        if b is None:
            self.b = self.add_param(
                lasagne.init.Constant(0), (num_filters,), name='b')
        else:
            self.b = self.add_param(
                b, (num_filters,), name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i * s - 2 * p + f - 1
                      for i, s, p, f in zip(input_shape[2:],
                                            self.stride,
                                            self.pad,
                                            self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)

def discriminator(input_var=None, dim_h=64, use_batch_norm=False, leak=0.01):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = LAYER_FN
    lrelu = LeakyRectify(leak)
    
    layer = InputLayer(shape=(None, N_CHANNELS, N_ROWS, N_COLS), input_var=input_var)
    
    layer = bn(Conv2DLayer(layer, dim_h, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = bn(Conv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = DenseLayer(layer, 1024, nonlinearity=lrelu)
    layer = DenseLayer(layer, 1, nonlinearity=linear)
    return layer

def generator(input_var=None, dim_z=100, dim_h=64, use_batch_norm=True):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = LAYER_FN
    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
    
    layer = bn(DenseLayer(layer, 1024))
    layer = bn(DenseLayer(layer, dim_h * 2 * 7 * 7))
    layer = ReshapeLayer(layer, ([0], dim_h * 2, 7, 7))
    layer = bn(Deconv2DLayer(layer, dim_h, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, N_CHANNELS, 5, stride=2, pad=2, nonlinearity=sigmoid)
    
    return layer

#################################### EXTRA MODELS #############################

# Low Capacity
def low_cap_discriminator(input_var=None, dim_h=64, use_batch_norm=False,
                        leak=0.01):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    lrelu = LeakyRectify(leak)
    
    layer = InputLayer(shape=(None, N_CHANNELS, N_ROWS, N_COLS), input_var=input_var)
    
    layer = bn(Conv2DLayer(layer, 8, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = bn(Conv2DLayer(layer, 8, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = DenseLayer(layer, 128, nonlinearity=lrelu)
    layer = DenseLayer(layer, 1, nonlinearity=linear)
    return layer

# High Capactiy
def high_cap_discriminator(input_var=None, dim_h=256, use_batch_norm=False,
                        leak=0.01):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    lrelu = LeakyRectify(leak)
    
    layer = InputLayer(shape=(None, N_CHANNELS, N_ROWS, N_COLS), input_var=input_var)
    
    layer = bn(Conv2DLayer(layer, dim_h, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = bn(Conv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = DenseLayer(layer, 4096, nonlinearity=lrelu)
    layer = DenseLayer(layer, 1, nonlinearity=linear)
    return layer
