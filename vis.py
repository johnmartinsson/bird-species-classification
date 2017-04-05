import os
import pickle
import itertools

from matplotlib import pyplot as plt
from sklearn import metrics
import configparser
import numpy as np

from bird import loader
import evaluate
import data_analysis

def chunks(l, n):
    chunk_size = int(np.ceil(len(l)/n))
    """Yield n chunks from l."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]

def plot_all(experiment_path):
    plot_confusion_matrix(experiment_path)
    plot_training_history(experiment_path)
    # plot_decending_training_samples_by_number_of_predictions(experiment_path,
                                                             # "datasets/birdClef2016Whole1/train/")

def load_predictions(experiment_path):
    picke_file = os.path.join(experiment_path, "predictions.pkl")
    print(picke_file)
    with open(picke_file, 'rb') as input:
        y_trues = pickle.load(input)
        y_scores = pickle.load(input)

    return y_trues, y_scores

def plot_decending_training_samples_by_number_of_predictions(experiment_path,
                                                             train_dir):
    config_parser = configparser.ConfigParser()
    config_parser.read(os.path.join(experiment_path, "conf.ini"))
    model_name = config_parser['MODEL']['ModelName']
    # order by number of training samples
    # compute number of predictions per class

    index_to_species = loader.build_class_index(train_dir)
    species_to_index = {v: k for k, v in index_to_species.items()}

    index_to_nb_training_segments = {}
    classes = os.listdir(train_dir)

    for c in classes:
        class_dir = os.path.join(train_dir, c)
        nb_training_segments = len(os.listdir(class_dir))
        index_to_nb_training_segments[species_to_index[c]] = nb_training_segments

    y_trues, y_scores = load_predictions(experiment_path)

    y_true = [np.argmax(y_t) for y_t in y_trues]
    y_pred = [np.argmax(y_s) for y_s in y_scores]
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    print(confusion_matrix)

    decending_by_training = []
    for key, value in index_to_nb_training_segments.items():
        decending_by_training.append((key, value))
    decending_by_training.sort(key=lambda x: x[1], reverse=True)

    ys1 = []
    ys2 = []
    ys3 = []

    for (key, value) in decending_by_training:
        actual_nb_predictions = np.sum(confusion_matrix[:,key])
        expected_nb_predictions = np.sum(confusion_matrix[key,:])
        ys1.append(actual_nb_predictions)
        ys2.append(expected_nb_predictions)
        ys3.append(value)

    ys1 = chunks(ys1, 20)
    ys1 = [np.mean(y) for y in ys1]

    ys2 = chunks(ys2, 20)
    ys2 = [np.mean(y) for y in ys2]

    ys3 = chunks(ys3, 20)
    ys3 = [np.mean(y) for y in ys3]

    title = "#Predictions and Training Segments ranked by #Training Segments"

    fig, ax1 = plt.subplots()
    plt.title(title)
    ax1.set_ylim([0, 10])
    ax1.plot(ys1, 'ro-', label="actual predictions")
    ax1.plot(ys2, 'go-', label="expected predictions")
    ax1.set_ylabel("Average Number of Predictions")
    ax1.set_xlabel("5% Chunks of Classes Ranked by Number of Training Segments (decreasing)")
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.set_ylim([0, 600])
    ax2.set_ylabel("Numer of Training Segments")
    ax2.plot(ys3, 'bo-', label="training segments")

    ax2.legend(loc="upper right")
    fig.savefig(os.path.join(experiment_path,
                             "training_samples_by_number_of_predictions.png"))
    fig.clf()
    plt.close(fig)

def plot_sound_class_by_decending_accuracy(experiment_path):
    config_parser = configparser.ConfigParser()
    config_parser.read(os.path.join(experiment_path, "conf.ini"))
    model_name = config_parser['MODEL']['ModelName']
    y_trues, y_scores = load_predictions(experiment_path)

    y_true = [np.argmax(y_t) for y_t in y_trues]
    y_pred = [np.argmax(y_s) for y_s in y_scores]

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    accuracies = []
    (nb_rows, nb_cols) = confusion_matrix.shape
    for i in range(nb_rows):
        accuracy = confusion_matrix[i][i] / np.sum(confusion_matrix[i,:])
        accuracies.append(accuracy)

    fig = plt.figure()
    plt.title("Sound Class ranked by Accuracy ({})".format(model_name))
    plt.plot(sorted(accuracies, reverse=True))
    plt.ylabel("Accuracy")
    plt.xlabel("Rank")
    # plt.pcolormesh(confusion_matrix, cmap=cmap)
    fig.savefig(os.path.join(experiment_path, "descending_accuracy.png"))


def print_top_n_confusions(top_n, experiment_path, train_dir):
    config_parser = configparser.ConfigParser()
    config_parser.read(os.path.join(experiment_path, "conf.ini"))
    model_name = config_parser['MODEL']['ModelName']

    index_to_species = loader.build_class_index(train_dir)

    y_trues, y_scores = load_predictions(experiment_path)

    y_true = [np.argmax(y_t) for y_t in y_trues]
    y_pred = [np.argmax(y_s) for y_s in y_scores]

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # confusion_matrix = np.flip(confusion_matrix, 0)

    (nb_rows, nb_cols) = confusion_matrix.shape

    confusions = []
    for i in range(nb_rows):
        confused_predictions = np.sum(confusion_matrix[:,i]) - confusion_matrix[i][i]
        confusions.append((i, confused_predictions))

    sorted_confusions = sorted(confusions, key=lambda x: x[1], reverse=True)
    for (i, confusion) in sorted_confusions[:top_n]:
        accuracy = confusion_matrix[i][i] / np.sum(confusion_matrix[i,:])
        print("Class {} ({}): {} (accuracy : {})".format(index_to_species[i],
                                                         i, confusion, accuracy))



def plot_confusion_matrix(experiment_path):
    config_parser = configparser.ConfigParser()
    config_parser.read(os.path.join(experiment_path, "conf.ini"))
    model_name = config_parser['MODEL']['ModelName']
    y_trues, y_scores = load_predictions(experiment_path)

    y_true = [np.argmax(y_t) for y_t in y_trues]
    y_pred = [np.argmax(y_s) for y_s in y_scores]

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # confusion_matrix = np.flip(confusion_matrix, 0)

    title = "Confusion Matrix ({})".format(model_name)

    cmap = plt.cm.get_cmap('jet')
    fig = plt.figure()
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    # plt.pcolormesh(confusion_matrix, cmap=cmap)
    fig.savefig(os.path.join(experiment_path, "confusion.png"))

    fig.clf()
    plt.close(fig)

    (nb_rows, nb_cols) = confusion_matrix.shape

    confusions = []
    for i in range(nb_rows):
        confused_predictions = np.sum(confusion_matrix[:,i]) - confusion_matrix[i][i]
        confusions.append((i, confused_predictions))

    sorted_confusions = sorted(confusions, key=lambda x: x[1], reverse=True)
    for (i, confusion) in sorted_confusions:
        print("Class {}: {}".format(i, confusion))

def plot_training_history(experiment_path):
    pickle_file = os.path.join(experiment_path, "history.pkl")
    with open(pickle_file, 'rb') as input:
        trainLoss = pickle.load(input)
        validLoss = pickle.load(input)
        trainAcc = pickle.load(input)
        validAcc = pickle.load(input)

        x = range(len(validAcc))
        y = validAcc
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)

        fig = plt.figure(1)
        plt.subplot(211)
        axes = plt.gca()
        axes.set_ylim([0, 10])
        plt.ylabel("Loss")
        plt.plot(trainLoss, 'o-', label="train")
        plt.plot(validLoss, 'o-', label="valid")
        plt.legend(loc="upper right")
        plt.subplot(212)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.plot(trainAcc, 'o-', label="train")
        plt.plot(validAcc, 'o-', label="valid")
        # plt.plot(p(x), 'o-', label="trend")
        plt.legend(loc="lower right")

        fig.savefig(os.path.join(experiment_path, "history.png"))
        plt.clf()
        plt.close()

# def plot_decending_accuracy_by_sound_class(experiment_path):
    # y_trues, y_scores = load_predictions(experiment_path)
    # y_labels = [np.argmax(y_t) for y_t in y_trues]
def main():
    plot_all()

if __name__ == "__main__":
    main()
