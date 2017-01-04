import numpy as np
from functools import reduce
from bird.models.cuberun import CubeRun
from bird.models.resnet import ResNetBuilder
from bird import utils
from bird import loader
from sklearn import metrics

import tqdm

nb_classes = 20
input_shape = (256, 512, 1)
batch_size=32

def evaluate(model, data_filepath):
    # (X_tests, Y_tests) = loader.load_test_data(data_filepath, file2labels_filepath,
                                               # nb_classes=nb_classes)
    (X_tests, Y_tests) = loader.load_test_data_birdclef(data_filepath,
                                                      input_shape)

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
    print("- Top 1:", top_1)
    print("- Top 2:", top_2)
    print("- Top 3:", top_3)
    print("- Top 4:", top_4)
    print("- Top 5:", top_5)
    print("")
    print("Mean Average Precision: ", np.mean(average_precision_scores))
    print("Area Under Curve: ", np.mean(roc_auc_scores))
    print("Total predictions: ", len(X_tests))

# model = CubeRun(nb_classes, input_shape)
model = ResNetBuilder.build_resnet_34(input_shape, nb_classes)
model.load_weights("./weights/2016_12_20_16:47:03_resnet.h5")
model.compile(loss="categorical_crossentropy", optimizer="adadelta")
evaluate(model, "./datasets/birdClef2016Subset/valid")

