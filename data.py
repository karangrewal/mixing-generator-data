import h5py
import numpy as np

def get_data(batch_size=32):
    path = '<path-to-mnist-hdf5-file'
    data = h5py.File(path)
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