import os
import numpy as np
import util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Model:

    def __init__(self, model):
        super(Model, self).__init__()

        self.model = model

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        predictions = []
        for row in input_array:
            input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
            predictions.append(np.argmax(self.model.predict(input).ravel()))
        return np.array(predictions)

    # Given a batch of examples return a batch of certainty levels.
    # predict gives vector of probabilities and display the max probability
    def certainty(self, input_array: np.ndarray) -> np.ndarray:
        certainties = []
        for row in input_array:
            input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
            prediction = self.model.predict(input).ravel()
            certainties.append(np.max(prediction))
        return np.array(certainties)
