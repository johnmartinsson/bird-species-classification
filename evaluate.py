import numpy as np
import pickle
from keras.applications.resnet50 import ResNet50
from functools import reduce
from bird.models.cuberun import CubeRun
from bird.models.resnet import ResNetBuilder
from bird import utils
from bird import loader
from sklearn import metrics
import os
import glob
import configparser
import ast
from optparse import OptionParser

import tqdm

parser = OptionParser()
parser.add_option("--experiment_path", dest="experiment_path")
(options, args) = parser.parse_args()

config_parser = configparser.ConfigParser()
config_parser.read(os.path.join(options.experiment_path, "conf.ini"))

validation = config_parser['PATHS']['ValidationDataDir']
train = config_parser['PATHS']['TrainingDataDir']

model_name = config_parser['MODEL']['ModelName']
weight_path = os.path.join(options.experiment_path, "weights.h5")
nb_classes = int(config_parser['MODEL']['NumberOfClasses'])
input_shape = ast.literal_eval(config_parser['MODEL']['InputShape'])
batch_size = int(config_parser['MODEL']['BatchSize'])

def evaluate(model, data_filepath):
    stats = {}


    # (X_tests, Y_tests) = loader.load_test_data(data_filepath, file2labels_filepath,
                                               # nb_classes=nb_classes)
    (X_tests, Y_tests, index_to_species) = loader.load_test_data_birdclef(data_filepath,
                                                                          input_shape)

    # init all species
    for i in range(nb_classes):
        stats[index_to_species[i]] = {
            "correct" : 0,
            "incorrect" : 0
        }

    for species in os.listdir(train):
        nb_valid = len(loader.group_segments(glob.glob(os.path.join(validation, species, "*.wav"))))
        nb_train = len(loader.group_segments(glob.glob(os.path.join(train, species, "*.wav"))))
        stats[species]["validation_samples"] = nb_valid
        stats[species]["training_samples"] = nb_train
        stats[species]["confusion"] = {}

    top_1 = 0
    top_2 = 0
    top_3 = 0
    top_4 = 0
    top_5 = 0
    average_precision_scores = []
    roc_auc_scores = []
    # print("| Predicted | Ground Truth |")
    # print("|-----------|--------------|")
    progress = tqdm.tqdm(range(len(X_tests)))
    for X_test, Y_test, p in zip(X_tests, Y_tests, progress):
        # print("X_test shape:", X_test.shape)
        # print("Y_test shape:", Y_test.shape)
        Y_preds = model.predict(X_test)
        y_score = np.mean(Y_preds, axis=0)
        y_pred = np.argmax(y_score)
        y_preds = np.argsort(y_score)[::-1]
        # print("y score", y_score)
        y_true = Y_test
        y_true_cat = np.argmax(y_true)

        if y_pred == y_true_cat:
            stats[index_to_species[y_true_cat]]["correct"] += 1
        else:
            stats[index_to_species[y_true_cat]]["incorrect"] += 1

        if not index_to_species[y_pred] in stats[index_to_species[y_true_cat]]["confusion"]:
            stats[index_to_species[y_true_cat]]["confusion"][index_to_species[y_pred]] = 1
        else:
            stats[index_to_species[y_true_cat]]["confusion"][index_to_species[y_pred]] += 1

        # compute average precision score
        average_precision_score = metrics.average_precision_score(y_true, y_score)
        average_precision_scores.append(average_precision_score)
        # compute roc auc score
        roc_auc_score = metrics.roc_auc_score(y_true, y_score)
        roc_auc_scores.append(roc_auc_score)

        if y_true_cat in y_preds[:1]:
            top_1+=1
        if y_true_cat in y_preds[:2]:
            top_2+=1
        if y_true_cat in y_preds[:3]:
            top_3+=1
        if y_true_cat in y_preds[:4]:
            top_4+=1
        if y_true_cat in y_preds[:5]:
            top_5+=1

        # print("| ", y_preds[:5], " | ", y_true_cat, " |")

    print("")
    print("- Top 1:", top_1/len(X_tests))
    print("- Top 2:", top_2/len(X_tests))
    print("- Top 3:", top_3/len(X_tests))
    print("- Top 4:", top_4/len(X_tests))
    print("- Top 5:", top_5/len(X_tests))
    print("")
    print("Mean Average Precision: ", np.mean(average_precision_scores))
    print("Area Under Curve: ", np.mean(roc_auc_scores))
    print("Total predictions: ", len(X_tests))

    with open(os.path.join(options.experiment_path, "evaluate.pkl"), "wb") as output:
        pickle.dump(stats, output, pickle.HIGHEST_PROTOCOL)

if model_name == "cuberun":
    model = CubeRun(nb_classes, input_shape)
elif model_name == "resnet_18":
    model = ResNetBuilder.build_resnet_18(input_shape, nb_classes)
elif model_name == "resnet_50":
    model = ResNet50(True, None, None, input_shape, nb_classes)
model.load_weights(weight_path)
model.compile(loss="categorical_crossentropy", optimizer="adadelta")
evaluate(model, validation)
