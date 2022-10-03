import os
import pandas as pd
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def make_model(num_classes=10, *args, **kwargs):
    # Create the model
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    skip = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Dropout(0.2)(skip)
    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    merge = keras.layers.add([x, skip])
    x = keras.layers.Activation('relu')(merge)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation='relu', kernel_constraint=maxnorm(3))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x, name="friends")
    print(model.summary())
    # Compile model
    lrate = 0.01
    decay = lrate / 100
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=decay, nesterov=False),
                  metrics=['accuracy'])
    return model


def predict(self, input_array: np.ndarray) -> np.ndarray:
    predictions = []
    for row in input_array:
        input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
        predictions.append(np.argmax(self.predict(input).ravel()))
    return np.array(predictions)


# Given a batch of examples return a batch of certainty levels.
# predict gives vector of probabilities and display the max probability
def certainty(self, input_array: np.ndarray) -> np.ndarray:
    certainties = []
    for row in input_array:
        input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
        prediction = self.predict(input).ravel()
        certainties.append(np.max(prediction))
    return np.array(certainties)
