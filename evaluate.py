import numpy as np
from functools import reduce
from bird.models.cuberun import CubeRun
from bird import utils
from bird import loader

nb_classes = 20
input_shape = (257, 512)
image_shape = input_shape
batch_size=32




def evaluate(model, data_filepath, file2labels_filepath):
    (X_tests, Y_tests) = loader.load_test_data(data_filepath, file2labels_filepath,
                                               nb_classes=nb_classes)

    top_1 = 0
    top_2 = 0
    top_3 = 0
    top_4 = 0
    top_5 = 0
    print("| Predicted | Ground Truth |")
    print("|-----------|--------------|")
    for X_test, Y_test in zip(X_tests, Y_tests):
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        Y_preds = model.predict(X_test)
        Y_mean = np.mean(Y_preds, axis=0)
        y_pred = np.argmax(Y_mean)
        y_preds = np.argsort(Y_mean)[::-1]
        y_true = np.argmax(Y_test[0])

        if y_true in y_preds[:1]:
            top_1+=1
        if y_true in y_preds[:2]:
            top_2+=1
        if y_true in y_preds[:3]:
            top_3+=1
        if y_true in y_preds[:4]:
            top_4+=1
        if y_true in y_preds[:5]:
            top_5+=1

        print("| ", y_preds[:5], " | ", y_true, " |")

    print("Top 1:", top_1)
    print("Top 2:", top_2)
    print("Top 3:", top_3)
    print("Top 4:", top_4)
    print("Top 5:", top_5)
    print("Total predictiosn: ", len(X_tests))

def to_str(x):
    if len(x) == 0:
        return ""
    else:
        ret = ""
        for s in x:
            ret = ret + ("," + str(s))
        return ret

def binary_to_id(Y):
    i = 0
    r = []
    for y in Y:
        if y == 1:
            r.append(i)
        i = i+1

    return r

model = CubeRun(nb_classes, input_shape)
model.load_weights("./weights/2016_11_30_14:13:35_cuberun.h5")
model.compile(loss="categorical_crossentropy", optimizer="adadelta")
evaluate(model, "./datasets/birdClef2016Subset_preprocessed/valid",
         "datasets/birdClef2016Subset_preprocessed/valid/file2labels.csv")

