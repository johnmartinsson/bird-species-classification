from models.cuberun import CubeRun
import numpy as np
import utils
import loader

nb_classes = 19
input_shape = (257, 509, 1)
(cols, rows, chs) = input_shape
image_shape = (cols, rows)
batch_size=32

def evaluate(model, data_filepath, file2labels_filepath):
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['fbeta_score'])
    (X_test, Y_test, filenames) = loader.load_all_data(data_filepath, file2labels_filepath,
                                            nb_classes=nb_classes,
                                            image_shape=image_shape)

    print("Predicting ...")
    Y = model.predict(X_test, batch_size=batch_size, verbose=1)
    Y = np.round(Y)
    print("| Predicted | Ground Truth |")
    print("|-----------|--------------|")
    for (y, gt) in zip(Y, Y_test):
        print("| ", binary_to_id(y), " | ", binary_to_id(gt), " |")

def binary_to_id(Y):
    i = 0
    r = []
    for y in Y:
        if y == 1:
            r.append(i)
        i = i+1

    return r
