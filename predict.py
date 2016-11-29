from models.cuberun import CubeRun
import numpy as np
import utils

def get_model():
    nb_classes = 19
    input_shape = (257, 624, 1)

    model = CubeRun(nb_classes=nb_classes, input_shape=input_shape)

    model.load_weights("../weights/2016_11_16_06:31:03_cuberun.h5")

    return model

def predict(model, filename):
    fs, x = utils.read_wave_file("../datasets/mlsp2013/test/"+filename+".wav")

    (f, t, Sxx) = utils.wave_to_spectrogram(x, fs)

    X = np.array([Sxx])

    X = X.reshape(X.shape[0], 257, 624, 1)

    y = model.predict(X, batch_size=32, verbose=1)
    y = np.round(y)

    return [binary_to_id(v) for v in  y]

def binary_to_id(Y):
    i = 0
    r = []
    for y in Y:
        if y == 1:
            r.append(i)
        i = i+1

    return r
