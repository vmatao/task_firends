import os
import pandas as pd
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.datasets import cifar10

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ModelFactory:

    def __init__(self, num_classes=10, epochs=10, batch_size=32):
        super(ModelFactory, self).__init__()

        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.inputs = keras.Input(shape=(32, 32, 3))
        self.kernel_size = (3, 3)
        self.activation = 'relu'
        self.padding = 'same'
        self.dropout = 0.2
        self.momentum = 0.9
        self.nesterov = False
        self.pool_size = (2, 2)
        self.lrate = 0.01
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
        skip = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(
            input)
        layer = keras.layers.Dropout(self.dropout)(skip)
        layer = keras.layers.Conv2D(filter, self.kernel_size, padding=self.padding)(layer)
        merge = keras.layers.add([layer, skip])
        activation = keras.layers.Activation('relu')(merge)
        return activation

    def make_main_block(self, input, filter):
        block = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(
            input)
        block = keras.layers.Dropout(self.dropout)(block)
        block = keras.layers.Conv2D(filter, self.kernel_size, activation=self.activation, padding=self.padding)(
            block)
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

    def train(self, model):
        # load data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # normalize inputs from 0-255 to 0.0-1.0
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, batch_size=self.batch_size)
        # model.save('proper_model_.h5')
        # print('Saved model to disk!')

    def create_and_train(self):
        model = self.make_model()
        print(model.summary())
        self.compile_model(model)
        self.train(model)
        return model