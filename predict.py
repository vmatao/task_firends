import os
import keras
import random
import pickle
import numpy as np
import cv2
import util
import model


def evaluate():
    # Load the model
    model_img = keras.models.load_model('models/resnet_50.h5')
    test_data, test_labels = util.load_test_data()

    predictions = model.predict(model_img, test_data[0:1000])

    certainties = model.certainty(model_img, test_data[0:1000])

    accuracy = model.accuracy(predictions,test_labels[0:1000])

    print(test_labels)
    print(predictions)
    print(certainties)
    print(accuracy)


def main():
    # Evaluate the model
    evaluate()


# Tell python to run main method
if __name__ == '__main__': main()
