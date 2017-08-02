import numpy as np
np.random.seed(42)

import gc
import csv
import os
import warnings
import random
import glob
import scipy

import librosa
import scipy

from bird import utils
from bird import data_augmentation as da
from bird import signal_processing as sp

def load_test_data_birdclef(directory, target_size, input_data_mode):
    if not os.path.isdir(directory):
        raise ValueError("data filepath is invalid")

    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    nb_classes = len(classes)
    class_indices = dict(zip(classes, range(nb_classes)))
    index_to_species = dict(zip(range(nb_classes), classes))

    X_test = []
    Y_test = []
    training_files = []
    for subdir in classes:
        subpath = os.path.join(directory, subdir)
        # load sound data
        class_segments = glob.glob(os.path.join(subpath, "*.wav"))
        # print(subdir+": ", len(class_segments))
        print("group segments ... ")
        samples = group_segments(class_segments)
        for sample in samples:
            training_files.append(sample)
            data = load_segments(sample, target_size, input_data_mode)
            X_test.append(data)
            y = np.zeros(nb_classes)
            y[class_indices[subdir]] = 1.0
            Y_test.append(y)
    return np.asarray(X_test), np.asarray(Y_test), training_files

def build_class_index(directory):
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    nb_classes = len(classes)
    class_indices = dict(zip(classes, range(nb_classes)))
    index_to_species = dict(zip(range(nb_classes), classes))
    return index_to_species



def load_segments(segments, target_size, input_data_mode):
    print(segments, target_size, input_data_mode)
    data = []
    for segment in segments:
        (fs, signal) = utils.read_wave_file(segment)
        if input_data_mode == "mfcc":
            sample = librosa.feature.mfcc(signal, fs, n_mfcc=target_size[0])
            sample = scipy.misc.imresize(sample, target_size)
            sample = sample.reshape((sample.shape[0],
                                     sample.shape[1], 1))
        if input_data_mode == "mfcc_delta":
            mfcc = librosa.feature.mfcc(signal, fs, n_mfcc=target_size[0])
            mfcc_delta_3 = librosa.feature.delta(mfcc, width=3, order=1)
            mfcc_delta_11 = librosa.feature.delta(mfcc, width=11, order=1)
            mfcc_delta_19 = librosa.feature.delta(mfcc, width=19, order=1)

            mfcc = scipy.misc.imresize(mfcc, target_size)
            mfcc_delta_3 = scipy.misc.imresize(mfcc_delta_3, target_size)
            mfcc_delta_11 = scipy.misc.imresize(mfcc_delta_11, target_size)
            mfcc_delta_19 = scipy.misc.imresize(mfcc_delta_19, target_size)

            mfcc = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)
            mfcc_delta_3 = mfcc_delta_3.reshape(mfcc_delta_3.shape[0], mfcc_delta_3.shape[1], 1)
            mfcc_delta_11 = mfcc_delta_11.reshape(mfcc_delta_11.shape[0], mfcc_delta_11.shape[1], 1)
            mfcc_delta_19 = mfcc_delta_19.reshape(mfcc_delta_19.shape[0], mfcc_delta_19.shape[1], 1)
            sample = np.concatenate([mfcc, mfcc_delta_3, mfcc_delta_11, mfcc_delta_19], axis=2)

        if input_data_mode == "spectrogram":
            sample = sp.wave_to_sample_spectrogram(signal, fs)
            sample = scipy.misc.imresize(sample, target_size)
            sample = sample.reshape((sample.shape[0],
                                     sample.shape[1], 1))
        data.append(sample)

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

