import h5py
import numpy as np

def get_data(batch_size=32):

    data = h5py.File('/u/grewalka/mnist/mnist.hdf5')
    data = data['features'][:]
    data = np.float32(data)
    data = data / 255.

    k = data.shape[0]
    k = k / batch_size
    k = k * batch_size
    k = data.shape[0] - k

    np.random.shuffle(data)
    data = data[k:]
    data = np.split(data, data.shape[0] / batch_size)
    data = np.array(data)

    return data