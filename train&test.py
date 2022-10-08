import os
from keras.utils import np_utils
import model as md
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

import util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(model, data):
    epochs = 1
    batch_size = 32

    # load data
    (x_train, y_train), (x_test, y_test) = data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125, random_state=1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    # normalizing inputs from 0-255 to 0.0-1.0
    x_train = x_train / 255.0
    x_val = x_val / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    num_classes = y_test.shape[1]

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)
    model.save("new_model.h5")


def evaluate(model, data):
    (x_train, y_train), (x_test, y_test) = data

    y_test = y_test.flatten()

    predictions, certainties, accuracy = util.all_in_one(model, x_test, y_test)

    # print(y_test)
    # print(predictions)
    # print(certainties)
    print(accuracy)


# The main entry point for this module
def main():
    data = cifar10.load_data()
    model = md.Model().build_model()
    train(model, data)
    # model_img = keras.models.load_model('new_model.h5')
    evaluate(model, data)


# Tell python to run main method
if __name__ == '__main__': main()
