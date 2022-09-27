# Import libraries
import os
import keras
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Get a ResNet50 model
def resnet50_model(classes=1000, *args, **kwargs):
    # Load a model if we have saved one
    if (os.path.isfile('/models/resnet_50.h5') == True):
        return keras.models.load_model('/models/resnet_50.h5')
    # Create an input layer
    input = keras.layers.Input(shape=(None, None, 3))
    print(input)
    # Create output layers
    output = keras.layers.ZeroPadding2D(padding=3, name='padding_conv1')(input)
    print(output)
    # stem
    output = stem(output, 3, [32, 32, 64])
    # output = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(output)
    # output = keras.layers.BatchNormalization(axis=3, epsilon=1e-5, name='bn_conv1')(output)
    # output = keras.layers.Activation('relu', name='conv1_relu')(output)
    # output = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(output)
    # Stage1-Block1
    output = conv_block(output, 3, [64, 64, 256], strides=(1, 1))
    # Stage1-Block2
    output = identity_block(output, 3, [64, 64, 256])
    output = identity_block(output, 3, [64, 64, 256])
    # FC-Block
    output = keras.layers.GlobalAveragePooling2D(name='pool5')(output)
    output = keras.layers.Dense(classes, activation='softmax', name='fc1000')(output)
    # Create a model from input layer and output layers
    model = keras.models.Model(inputs=input, outputs=output, *args, **kwargs)
    # Print model
    print()
    print(model.summary(), '\n')
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.001),
                  metrics=['accuracy'])
    # Return a model
    return model


def stem(input, kernel_size, filters, strides=(2, 2)):
    # Variables
    filters1, filters2, filters3 = filters
    output = keras.layers.Conv2D(filters1, kernel_size, strides=strides, kernel_initializer='he_normal')(input)
    output = keras.layers.Conv2D(filters2, kernel_size, kernel_initializer='he_normal')(output)
    output = keras.layers.Conv2D(filters3, kernel_size, kernel_initializer='he_normal')(output)
    output = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(output)
    return output


# Create an identity block
def identity_block(input, kernel_size, filters):
    filters1, filters2, filters3 = filters
    output = keras.layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal')(input)
    output = keras.layers.BatchNormalization(axis=3)(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal')(output)
    output = keras.layers.BatchNormalization(axis=3)(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal')(output)
    output = keras.layers.BatchNormalization(axis=3)(output)
    output = keras.layers.add([output, input])
    output = keras.layers.Activation('relu')(output)
    return output


# Create a convolution block
def conv_block(input, kernel_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    output = keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal')(input)
    output = keras.layers.BatchNormalization(axis=3)(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal')(output)
    output = keras.layers.BatchNormalization(axis=3)(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal')(output)
    output = keras.layers.BatchNormalization(axis=3)(output)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal')(input)
    shortcut = keras.layers.BatchNormalization(axis=3)(shortcut)
    output = keras.layers.add([output, shortcut])
    output = keras.layers.Activation('relu')(output)
    return output


# Given a batch of examples return a batch of predicted classes.
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


def accuracy(preds, labels):
    dict_acc_ev = {}
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    correct = 0
    total = 0
    result = 0
    total += np.size(labels)
    correct += (preds == labels).sum().item()
    # if total != 0:
    result = 100 * correct / total

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # collect the correct predictions for each class
    for label, prediction in zip(labels, preds):
        if label == prediction:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            class_acc = 100 * float(correct_count) / total_pred[classname]
            dict_acc_ev[classname] = class_acc

    df_class_accuracy = pd.DataFrame.from_dict(dict_acc_ev, orient='index')
    df_class_accuracy = df_class_accuracy.T
    return result, df_class_accuracy
