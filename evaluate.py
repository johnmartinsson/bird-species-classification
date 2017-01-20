import numpy as np
import pickle
from functools import reduce
from bird.models.cuberun import CubeRun
from bird.models.resnet import ResNetBuilder
from bird import utils
from bird import loader
from sklearn import metrics
import os
import glob
from optparse import OptionParser

import tqdm

parser = OptionParser()
parser.add_option("--valid_path", dest="valid_path")
parser.add_option("--train_path", dest="train_path")
(options, args) = parser.parse_args()

validation = options.valid_path
train = options.train_path

def chunks(l, n):
    chunk_size = int(np.ceil(len(l)/n))
    """Yield n chunks from l."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]

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

    with open("test.pkl", "wb") as output:
        pickle.dump(stats, output, pickle.HIGHEST_PROTOCOL)

nb_classes = 809
input_shape = (256, 512, 1)
batch_size=32

# model = CubeRun(nb_classes, input_shape)
model = ResNetBuilder.build_resnet_18(input_shape, nb_classes)
model.load_weights("./weights/2017_01_18_19:27:53_resnet_18.h5")
model.compile(loss="categorical_crossentropy", optimizer="adadelta")
evaluate(model, validation)
