import keras
import util
import model


def evaluate():
    # Load the model
    model_img = keras.models.load_model('models/resnet_50.h5')
    test_data, test_labels = util.load_test_data()

    predictions, certainties, accuracy = model.all_in_one(model_img, test_data, test_labels)

    # predictions = model.predict(model_img, test_data)
    #
    # certainties = model.certainty(model_img, test_data)
    #
    # accuracy = model.accuracy(predictions, test_labels)
    #
    # model.create_historigram(certainties, predictions, test_labels)

    print(test_labels)
    print(predictions)
    print(certainties)
    print(accuracy)


def main():
    # Evaluate the model
    evaluate()


# Tell python to run main method
if __name__ == '__main__': main()
