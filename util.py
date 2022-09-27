import pickle

data_file = 'cifar-10-batches-py/'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data():
    data_list = []
    for i in range(1, 6):
        data_list.append(unpickle(data_file+'/train/data_batch_' + str(i)))
    return data_list


def load_test_data():
    return unpickle(data_file+'/test/test_batch')
