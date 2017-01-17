import numpy as np
np.random.seed(42)

import gc
import csv
import os
import warnings
import random
import glob
import scipy

from bird import utils
from bird import data_augmentation as da
from bird import signal_processing as sp

def load_test_data_birdclef(directory, target_size):
    if not os.path.isdir(directory):
        raise ValueError("data filepath is invalid")

    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    nb_classes = len(classes)
    class_indices = dict(zip(classes, range(nb_classes)))

    X_test = []
    Y_test = []
    for subdir in classes:
        subpath = os.path.join(directory, subdir)
        # load sound data
        class_segments = glob.glob(os.path.join(subpath, "*.wav"))
        print(subdir+": ", len(class_segments))
        samples = group_segments(class_segments)
        for sample in samples:
            data = load_segments(sample, target_size)
            X_test.append(data)
            y = np.zeros(nb_classes)
            y[class_indices[subdir]] = 1.0
            Y_test.append(y)
    return np.asarray(X_test), np.asarray(Y_test)

def load_segments(segments, target_size):
    data = []
    for segment in segments:
        (fs, signal) = utils.read_wave_file(segment)
        spectrogram = sp.wave_to_sample_spectrogram(signal, fs)
        spectrogram = scipy.misc.imresize(spectrogram, target_size)
        spectrogram = spectrogram.reshape((spectrogram.shape[0],
                                           spectrogram.shape[1], 1))
        data.append(spectrogram)

    return np.asarray(data)

def group_segments(segments):
    unique_samples = []
    for segment in segments:
        splits = segment.split('_')
        if not splits[4] in unique_samples:
            unique_samples.append(splits[4])

    samples = []
    for unique_sample in unique_samples:
        sample = []
        for segment in segments:
            if segment.split('_')[4] == unique_sample:
                sample.append(segment)
        # print(unique_sample, ":", len(sample))
        samples.append(sample)
    return samples

def id_labels2binary_labels(labels, nb_classes):
    """ Convert categorical labels to binary labels
    """
    binary_labels = np.zeros(nb_classes)
    for l in labels:
        binary_labels[l] = 1

    return np.array(binary_labels)

