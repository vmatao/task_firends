import os
import pandas as pd
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import model
from keras.datasets import cifar10
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train():
    # let's load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalizing inputs from 0-255 to 0.0-1.0
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    model_img = model.make_model(num_classes=10)
    model_img.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    model_img.save('proper_model.h5')
    print('Saved model to disk!')


# The main entry point for this module
def main():
    # Train a model
    train()


# Tell python to run main method
if __name__ == '__main__': main()
