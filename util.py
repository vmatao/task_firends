import pickle
import numpy as np

data_file = 'cifar-10-batches-py/'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data():
    data, labels = [], []
    for i in range(1, 6):
        dict_file = unpickle(data_file+'/train/data_batch_' + str(i))
        labels.extend(dict_file[b'labels'])
        data.extend(dict_file[b'data'])
    return np.array(data), np.array(labels)


def load_test_data():
    return unpickle(data_file+'/test/test_batch')
