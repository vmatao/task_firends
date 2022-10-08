import os
import pandas as pd
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Model:

    def __init__(self, num_classes=10, activation = 'relu', padding = 'same', dropout = 0.2,
                 momentum = 0.9, nesterov = False, shape=(32, 32, 3), pool_size = (2, 2), lrate = 0.01, kernel_size = (3, 3)):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.inputs = keras.Input(shape=shape)
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.momentum = momentum
        self.nesterov = nesterov
        self.pool_size = pool_size
        self.lrate = lrate
        self.decay = self.lrate / 100

    def make_stem(self, filters=[32, 32, 64], shape=(32, 32, 3)):
        filter1, filter2, filter3 = filters
        stem = keras.layers.Conv2D(filter1, self.kernel_size, input_shape=shape, activation=self.activation,
                                   padding=self.padding)(self.inputs)
        stem = keras.layers.Conv2D(filter2, self.kernel_size, input_shape=shape, activation=self.activation,
                                   padding=self.padding)(stem)
        stem = keras.layers.Conv2D(filter3, self.kernel_size, input_shape=shape, activation=self.activation,
                                   padding=self.padding)(stem)
        stem = keras.layers.MaxPooling2D(self.kernel_size, strides=(2, 2), padding=self.padding)(stem)
        return stem

    def make_skip_connection(self, input, filter=64):
        skip = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(input)
        layer = keras.layers.Dropout(self.dropout)(skip)
        layer = keras.layers.Conv2D(filter, self.kernel_size, padding=self.padding)(layer)
        merge = keras.layers.add([layer, skip])
        activation = keras.layers.Activation('relu')(merge)
        return activation

    def make_main_block(self, input, filter):
        block = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(input)
        block = keras.layers.Dropout(self.dropout)(block)
        block = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(block)
        block = keras.layers.MaxPooling2D(pool_size=self.pool_size)(block)
        return block

    def make_dense_dropout(self, input, filter, kernel_constraint=maxnorm(3)):
        dense = keras.layers.Dense(filter, activation=self.activation, kernel_constraint=kernel_constraint)(input)
        dropout = keras.layers.Dropout(self.dropout)(dense)
        return dropout

    def make_model(self):
        output = self.make_stem()

        output = self.make_skip_connection(output)

        output = keras.layers.MaxPooling2D(pool_size=self.pool_size)(output)

        output = self.make_main_block(output, 128)
        output = self.make_main_block(output, 256)

        output = keras.layers.Flatten()(output)
        output = keras.layers.Dropout(self.dropout)(output)

        output = self.make_dense_dropout(output, 1024)
        output = self.make_dense_dropout(output, 512)

        output = keras.layers.Dense(self.num_classes, activation='softmax')(output)

        return keras.Model(inputs=self.inputs, outputs=output)

    def compile_model(self, model):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.lrate,
                                            momentum=self.momentum,
                                            decay=self.decay,
                                            nesterov=self.nesterov)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    def build_model(self):
        model = self.make_model()
        print(model.summary())
        self.compile_model(model)
        return model

    def predict(self, model, input_array: np.ndarray) -> np.ndarray:
        predictions = []
        for row in input_array:
            input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
            predictions.append(np.argmax(model.predict(input).ravel()))
        return np.array(predictions)

    # Given a batch of examples return a batch of certainty levels.
    # predict gives vector of probabilities and display the max probability
    def certainty(self, model, input_array: np.ndarray) -> np.ndarray:
        certainties = []
        for row in input_array:
            input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
            prediction = model.predict(input).ravel()
            certainties.append(np.max(prediction))
        return np.array(certainties)
