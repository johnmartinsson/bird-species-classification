import numpy as np
import pickle
from keras.applications.resnet50 import ResNet50
from functools import reduce
from bird.models.cuberun import CubeRun
from bird.models.resnet import ResNetBuilder
from bird import utils
from bird import loader
# from sklearn import metrics
import sklearn
import os
import glob
import configparser
import ast
from optparse import OptionParser

import tqdm

# parser = OptionParser()
# parser.add_option("--experiment_path", dest="experiment_path")
# (options, args) = parser.parse_args()

# config_parser = configparser.ConfigParser()
# config_parser.read(os.path.join(options.experiment_path, "conf.ini"))

# validation = config_parser['PATHS']['ValidationDataDir']
# train = config_parser['PATHS']['TrainingDataDir']

# model_name = config_parser['MODEL']['ModelName']
# weight_path = os.path.join(options.experiment_path, "weights.h5")
# nb_classes = int(config_parser['MODEL']['NumberOfClasses'])
# input_shape = ast.literal_eval(config_parser['MODEL']['InputShape'])
# batch_size = int(config_parser['MODEL']['BatchSize'])

def average_prediction(model, X_tests):
    y_scores = model.predict(X_tests)
    y_average_score = np.mean(y_scores, axis=0)
    return y_average_score

def top_n(y_trues, y_scores, n):
    score = 0
    for y_t, y_s in zip(y_trues, y_scores):
        top = np.argsort(y_s)[::-1]
        y = np.argmax(y_t)
        if y in top[:n]:
            score += 1
    return score/len(y_scores)

def mean_average_precision(y_trues, y_scores):
    """
    y_trues  : [nb_samples, nb_classes]
    y_scores : [nb_samples, nb_classes]

    map      : float (MAP)
    """
    aps = []
    for y_t, y_s in zip(y_trues, y_scores):
        ap = sklearn.metrics.average_precision_score(y_t, y_s)
        aps.append(ap)
    return np.mean(np.array(aps))

def area_under_roc_curve(y_trues, y_scores):
    """
    y_trues  : [nb_samples, nb_classes]
    y_scores : [nb_samples, nb_classes]

    map      : float (AUROC)
    """
    auroc = sklearn.metrics.roc_auc_score(y_trues, y_scores)
    return auroc

def evaluate(experiment_path):
    pickle_path = os.path.join(experiment_path, "predictions.pkl")
    with open(pickle_path, 'rb') as input:
        y_trues = pickle.load(input)
        y_scores = pickle.load(input)

    map_score = mean_average_precision(y_trues, y_scores)
    auroc_score = area_under_roc_curve(y_trues, y_scores)

    print("")
    print("- Top 1:", top_n(y_trues, y_scores, 1))
    print("- Top 2:", top_n(y_trues, y_scores, 2))
    print("- Top 3:", top_n(y_trues, y_scores, 3))
    print("- Top 4:", top_n(y_trues, y_scores, 4))
    print("- Top 5:", top_n(y_trues, y_scores, 5))
    print("")
    print("Mean Average Precision: ", map_score)
    print("Area Under Curve: ", auroc_score)
    print("Total predictions: ", len(y_scores))

def main():
    parser = OptionParser()
    parser.add_option("--experiment_path", dest="experiment_path")
    (options, args) = parser.parse_args()
    evaluate(options.experiment_path)

if __name__ == "main()":
    main()

# def evaluate(model, data_filepath):
    # print("Running new...")
    # stats = {}


    # (X_tests, Y_tests, index_to_species) = loader.load_test_data_birdclef(data_filepath,
                                                                          # input_shape)
    # # init all species
    # for i in range(nb_classes):
        # stats[index_to_species[i]] = {
            # "correct" : 0,
            # "incorrect" : 0
        # }

    # for species in os.listdir(train):
        # nb_valid = len(loader.group_segments(glob.glob(os.path.join(validation, species, "*.wav"))))
        # nb_train = len(loader.group_segments(glob.glob(os.path.join(train, species, "*.wav"))))
        # stats[species]["validation_samples"] = nb_valid
        # stats[species]["training_samples"] = nb_train
        # stats[species]["confusion"] = {}

    # y_scores = []
    # y_trues = Y_tests
    # progress = tqdm.tqdm(range(len(X_tests)))
    # for X_test, Y_test, p in zip(X_tests, Y_tests, progress):
        # y_score = average_prediction(model, X_test)
        # y_scores.append(y_score)

        # y_pred = np.argmax(y_score)
        # y_preds = np.argsort(y_score)[::-1]
        # y_true = Y_test
        # y_true_cat = np.argmax(y_true)

        # if y_pred == y_true_cat:
            # stats[index_to_species[y_true_cat]]["correct"] += 1
        # else:
            # stats[index_to_species[y_true_cat]]["incorrect"] += 1

        # if not index_to_species[y_pred] in stats[index_to_species[y_true_cat]]["confusion"]:
            # stats[index_to_species[y_true_cat]]["confusion"][index_to_species[y_pred]] = 1
        # else:
            # stats[index_to_species[y_true_cat]]["confusion"][index_to_species[y_pred]] += 1

    # map_score = mean_average_precision(y_trues, y_scores)
    # auroc_score = area_under_roc_curve(y_trues, y_scores)

    # print("")
    # print("- Top 1:", top_n(y_trues, y_scores, 1))
    # print("- Top 2:", top_n(y_trues, y_scores, 2))
    # print("- Top 3:", top_n(y_trues, y_scores, 3))
    # print("- Top 4:", top_n(y_trues, y_scores, 4))
    # print("- Top 5:", top_n(y_trues, y_scores, 5))
    # print("")
    # print("Mean Average Precision: ", map_score)
    # print("Area Under Curve: ", auroc_score)
    # print("Total predictions: ", len(X_tests))

    # with open(os.path.join(options.experiment_path, "evaluate.pkl"), "wb") as output:
        # pickle.dump(stats, output, pickle.HIGHEST_PROTOCOL)

# if model_name == "cuberun":
    # model = CubeRun(nb_classes, input_shape)
# elif model_name == "resnet_18":
    # model = ResNetBuilder.build_resnet_18(input_shape, nb_classes)
# elif model_name == "resnet_50":
    # model = ResNet50(True, None, None, input_shape, nb_classes)
# model.load_weights(weight_path)
# model.compile(loss="categorical_crossentropy", optimizer="adadelta")
# evaluate(model, validation)
