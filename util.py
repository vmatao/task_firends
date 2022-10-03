import pickle
import numpy as np
import pandas as pd

data_file = 'cifar-10-batches-py/'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data():
    data, labels = [], []
    for i in range(1, 6):
        dict_file = unpickle(data_file+'/train/data_batch_' + str(i))
        labels.extend(dict_file[b'labels'])
        data.extend(dict_file[b'data'])
    return np.array(data), np.array(labels)


def load_test_data():
    data, labels = [], []
    dict_file = unpickle(data_file + '/test/test_batch')
    labels.extend(dict_file[b'labels'])
    data.extend(dict_file[b'data'])
    return np.array(data), np.array(labels)

def all_in_one(self, input_array: np.ndarray, labels):
    predictions = []
    certainties = []
    for row in input_array:
        input = np.array(row).reshape((1, 32, 32, 3)).astype('float32') / 255
        prediction = self.predict(input).ravel()
        certainties.append(np.max(prediction))
        predictions.append(np.argmax(prediction))

    predictions = np.array(predictions)
    certainties = np.array(certainties)
    accuracy_o = accuracy(predictions, labels)
    create_historigram(certainties, predictions, labels)
    return predictions, certainties, accuracy_o


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


def create_historigram(certainties, predictions, labels):
    # fig, axis = plt.subplots(figsize=(10, 5))
    predictions_unique = np.unique(predictions)
    df_pred_correct = pd.DataFrame()
    df_pred_wrong = pd.DataFrame()
    df_pred_correct_after = pd.DataFrame()
    df_pred_wrong_after = pd.DataFrame()
    for i in range(0, certainties.size):
        if predictions[i] == labels[i]:
            temp_df = pd.DataFrame([[certainties[i]]], columns=["c " + str(labels[i])])
            df_pred_correct = pd.concat([df_pred_correct, temp_df])
            temp_df = pd.DataFrame([[certainties[i]]], columns=[str(labels[i]) + "c "])
            df_pred_correct_after = pd.concat([df_pred_correct_after, temp_df])
        else:
            temp_df = pd.DataFrame([[certainties[i]]], columns=["w " + str(predictions[i])])
            df_pred_wrong = pd.concat([df_pred_wrong, temp_df])
            temp_df = pd.DataFrame([[certainties[i]]], columns=[str(labels[i]) + "w "])
            df_pred_wrong_after = pd.concat([df_pred_wrong_after, temp_df])
    df = pd.concat([df_pred_correct, df_pred_wrong])
    df = pd.DataFrame(df.mean(axis=0), columns=['Certainty'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'predictions'})
    df.sort_values('predictions').plot.bar(x="predictions", y="Certainty", rot=70)
    plt.show(block=True)

    df = pd.concat([df_pred_correct_after, df_pred_wrong_after])
    df = pd.DataFrame(df.mean(axis=0), columns=['Certainty'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'predictions'})
    df.sort_values('predictions').plot.bar(x="predictions", y="Certainty", rot=70)
    plt.show(block=True)

    df_pred_correct = df_pred_correct[sorted(df_pred_correct.columns)]
    df_pred_correct.hist(sharex=True, figsize=(20, 10))
    plt.show(block=True)
    df_pred_wrong = df_pred_wrong[sorted(df_pred_wrong.columns)]
    df_pred_wrong.hist(sharex=True, figsize=(20, 10))
    plt.show(block=True)
