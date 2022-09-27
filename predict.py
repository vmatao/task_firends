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
    model_img = keras.models.load_model('C:\\Users\\vladm\\Desktop\\task_firends\\models\\resnet_50.h5')
    test_data = util.load_test_data()
    test_labels = test_data[b'labels']
    test_data = test_data[b'data']

    predictions = model.predict(model_img, test_data)

    # certainties = model.certainty(model_img, test_data)

    accuracy = model.accuracy(predictions,test_labels)

    print(test_labels)
    print(predictions)
    # print(certainties)
    print(accuracy)


def main():
    # Evaluate the model
    evaluate()


# Tell python to run main method
if __name__ == '__main__': main()
