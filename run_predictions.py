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

def average_prediction(model, X_tests):
    y_scores = model.predict(X_tests)
    y_average_score = np.mean(y_scores, axis=0)
    return y_average_score

def main():
    parser = OptionParser()
    parser.add_option("--experiment_path", dest="experiment_path")
    parser.add_option("--test_data", dest="test_data")
    (options, args) = parser.parse_args()

    config_parser = configparser.ConfigParser()
    config_parser.read(os.path.join(options.experiment_path, "conf.ini"))

    validation_dir = ""
    if not options.test_data:
        validation_dir = config_parser['PATHS']['ValidationDataDir']
    else:
        validation_dir = options.test_data

    model_name = config_parser['MODEL']['ModelName']
    weight_path = os.path.join(options.experiment_path, "weights.h5")
    nb_classes = int(config_parser['MODEL']['NumberOfClasses'])
    input_shape = ast.literal_eval(config_parser['MODEL']['InputShape'])
    input_data_mode = config_parser['MODEL']['InputDataMode']
    batch_size = int(config_parser['MODEL']['BatchSize'])

    if model_name == "cuberun":
        model = CubeRun(nb_classes, input_shape)
    elif model_name == "resnet_18":
        model = ResNetBuilder.build_resnet_18(input_shape, nb_classes)

    model.load_weights(weight_path)
    model.compile(loss="categorical_crossentropy", optimizer="adadelta")

    print("loading test data ... ")
    (X_tests, Y_tests, training_files) = loader.load_test_data_birdclef(validation_dir,
                                                                          input_shape, input_data_mode)
    y_scores = []
    y_trues = Y_tests
    progress = tqdm.tqdm(range(len(X_tests)))
    print("running predictions ... ")
    for X_test, Y_test, p in zip(X_tests, Y_tests, progress):
        y_score = average_prediction(model, X_test)
        y_scores.append(y_score)

    with open(os.path.join(options.experiment_path, "predictions.pkl"), "wb") as output:
        pickle.dump(y_trues, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_scores, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(training_files, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
