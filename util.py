import numpy as np
import pandas as pd

from charts import Charts


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

def all_in_one(model, input_array: np.ndarray, labels):
    predictions = []
    certainties = []
    for row in input_array:
        input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
        prediction = model.predict(input).ravel()
        certainties.append(np.max(prediction))
        predictions.append(np.argmax(prediction))

    predictions = np.array(predictions)
    certainties = np.array(certainties)
    accuracy_o = accuracy(predictions, labels)
    charts = Charts(certainties, predictions, labels)
    charts.plot_barcharts()
    charts.plot_histograms()
    return predictions, certainties, accuracy_o