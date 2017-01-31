from bird import utils
from bird.models.cuberun import CubeRun
import bird.loader as loader
import bird.signal_processing as sp
import scipy

import numpy as np

def predict(model, segment_names, directory):
    class_index = loader.build_class_index(directory)
    batch = []
    for segment_name in segment_names:
        # load input data
        fs, wave = utils.read_wave_file(segment_name)
        Sxx = sp.wave_to_sample_spectrogram(wave, fs)
        Sxx = scipy.misc.imresize(Sxx, (256, 512))
        batch.append(Sxx)
    batch = np.array(batch)

    batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2], 1)

    y_probs = model.predict(batch, batch_size=16, verbose=1)
    y_cats = [int(np.argmax(y_prob)) for y_prob in y_probs]
    species = [class_index[y_cat] for y_cat in y_cats]

    return species

def binary_to_id(Y):
    i = 0
    r = []
    for y in Y:
        if y == 1:
            r.append(i)
        i = i+1

    return r
