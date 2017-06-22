import h5py
import numpy as np
import pickle

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

    k = k / batch_size
    k = k * batch_size
    k = data.shape[0] - k

    np.random.shuffle(data)
    data = data[k:]
    data = np.split(data, data.shape[0] / batch_size)
    data = np.array(data)

    return data
