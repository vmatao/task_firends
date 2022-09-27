import os
import keras
import random
import pickle
import numpy as np
import cv2


# Evaluate the model
def evaluate():
    # Load the model
    model = keras.models.load_model('models/resnet_50.h5')
    # Load classes
    classes = {}
    with open('C:\\Users\\vladm\\Desktop\\task_firends\\CIFAR-10-images-master\\classes.pkl', 'rb') as file:
        classes = pickle.load(file)
    # Get a list of categories
    categories = os.listdir('C:\\Users\\vladm\\Desktop\\task_firends\\CIFAR-10-images-master\\test')
    # Get a category a random
    sum_corr = 0
    total = 0
    result = 0
    for category in categories:
        # Print the category
        print(category)
        # Get images in a category
        images = os.listdir('C:\\Users\\vladm\\Desktop\\task_firends\\CIFAR-10-images-master\\test\\' + category)
        # Randomize images to get different images each time
        random.shuffle(images)
        # Loop images
        blocks = []
        for i, name in enumerate(images):
            # Limit the evaluation
            if i > 100:
                break
            # Print the name
            print(name)
            # Get the image
            image = cv2.imread('C:\\Users\\vladm\\Desktop\\task_firends\\CIFAR-10-images-master\\test\\' + category + '\\' + name)
            # Get input reshaped and rescaled
            input = np.array(image).reshape((1, 32, 32, 3)).astype('float32') / 255
            # Get predictions
            predictions = model.predict(input).ravel()
            # Print predictions
            print(predictions)
            # Get the class with the highest probability
            prediction = np.argmax(predictions)
            # Check if the prediction is correct
            sum_corr += 1 if classes[prediction].lower() == category else 0
            total += 1
            # if total != 0:
    result = 100 * sum_corr / total
    print(result)

        # Draw the image and show the best prediction
        # image = cv2.resize(image, (256, 256))
        # cv2.putText(image, '{0}: {1} %'.format(classes[prediction], str(round(predictions[prediction] * 100, 2))),
        #             (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        # cv2.putText(image, '{0}: {1} %'.format(classes[prediction], str(round(predictions[prediction] * 100, 2))),
        #             (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (65, 105, 225), 2)
        # cv2.putText(image, '{0}'.format('CORRECT!' if correct else 'WRONG!'), (12, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
        #             (0, 0, 0), 2)
        # cv2.putText(image, '{0}'.format('CORRECT!' if correct else 'WRONG!'), (10, 48), cv2.FONT_HERSHEY_DUPLEX, 0.7,
        #             (0, 255, 0) if correct else (0, 0, 255), 2)

        # Append the image
    #     blocks.append(image)
    #
    # # Display images and predictions
    # row1 = np.concatenate(blocks[0:3], axis=1)
    # row2 = np.concatenate(blocks[3:6], axis=1)
    # # cv2.imshow('Predictions', np.concatenate((row1, row2), axis=0))
    # cv2.imwrite('C:\\Users\\vladm\\Desktop\\task_firends\\CIFAR-10-images-master\\plots\\predictions.jpg', np.concatenate((row1, row2), axis=0))
    # cv2.waitKey(0)


# The main entry point for this module
def main():
    # Evaluate the model
    evaluate()


# Tell python to run main method
if __name__ == '__main__': main()