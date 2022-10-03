import keras
import util
from keras.datasets import cifar10


def evaluate():
    # Load the model
    model_img = keras.models.load_model('project_model.h5')
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    predictions, certainties, accuracy = util.all_in_one(model_img, X_test, y_test)
    # predictions, certainties, accuracy = sf_task.all_in_one(model_img, test_data[0:5], test_labels[0:5])

    # predictions = model.predict(model_img, test_data)
    #
    # certainties = model.certainty(model_img, test_data)
    #
    # accuracy = model.accuracy(predictions, test_labels)
    #
    # model.create_historigram(certainties, predictions, test_labels)

    print(y_test)
    print(predictions)
    print(certainties)
    print(accuracy)


def main():
    # Evaluate the model
    evaluate()


# Tell python to run main method
if __name__ == '__main__': main()
