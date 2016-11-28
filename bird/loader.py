import numpy as np
np.random.seed(42)

import gc
import csv
import os
import warnings
import random
import glob

from bird import utils
from bird import data_augmentation as da

def mini_batch_generator(nb_augmentation_samples, nb_mini_baches, batch_size,
                         data_filepath, file2labels_filepath, nb_classes,
                         samplerate):
    i_batch_counter = 0
    augmentation_set = create_augmentation_set(data_filepath,
                                               file2labels_filepath,
                                               nb_augmentation_samples)
    while i_batch_counter < nb_mini_baches:
        mini_batch = np.random.choice(augmentation_set, batch_size)
        signals_and_labels = [da.apply_augmentation(d) for d in mini_batch]
        # extract labels
        Y_train_aux = [l for (s, l) in signals_and_labels]
        # convert to waves
        X_train_aux = [s for (s, l) in signals_and_labels]
        # convert to spectrograms
        X_train = [utils.wave_to_spectrogram_aux(s, samplerate) for s in X_train_aux]
        # convert to binary labels
        Y_train = [id_labels2binary_labels(l, nb_classes) for l in Y_train_aux]
        # force the garbage collector to clear unreferenced memory
        gc.collect()
        Y_train = np.array(Y_train)
        X_train = np.array(X_train)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                                            X_train.shape[2], 1)
        yield  X_train, Y_train
        i_batch_counter += 1

def load_noise_segment_filenames(data_filepath):
    noise_segment_filenames = [utils.get_basename_without_ext(f) for f in
                           glob.glob(os.path.join(data_filepath, "noise",
                                                  "*.wav.gz"))]
    return noise_segment_filenames

def load_signal_segment_filenames(data_filepath):
    signal_segment_filenames = [utils.get_basename_without_ext(f) for f in
                            glob.glob(os.path.join(data_filepath, "*.wav.gz"))]
    return signal_segment_filenames


def create_augmentation_set(data_filepath, file2labels_filepath, nb_augmentation_dicts):
    signal_segment_filenames = load_signal_segment_filenames(data_filepath)
    noise_segment_filenames = load_noise_segment_filenames(data_filepath)
    file2labels = read_file2labels(file2labels_filepath)

    augmentation_set = []
    for i in range(nb_augmentation_dicts):
        signal_segment_filename = np.random.choice(signal_segment_filenames, 1)[0]
        augmentation_dict = da.create_augmentation_dict(signal_segment_filename,
                                                        noise_segment_filenames, file2labels)
        augmentation_dict = reconstruct_relative_paths(augmentation_dict, data_filepath)
        augmentation_set.append(augmentation_dict)

    return augmentation_set

def reconstruct_relative_paths(augmentation_dict, data_filepath):
    signal_filename = augmentation_dict['signal_filename']
    augmentation_signal_filename = augmentation_dict['augmentation_signal_filename']
    augmentation_noise_filenames = augmentation_dict['augmentation_noise_filenames']
    labels = augmentation_dict['labels']

    signal_filepath = os.path.join(data_filepath, signal_filename+".wav.gz")
    augmentation_signal_filepath = os.path.join(data_filepath,
                                                augmentation_signal_filename
                                                +".wav.gz")
    augmentation_noise_filepaths = [os.path.join(data_filepath, "noise",
                                                 noise_filename+".wav.gz") for
                                                 noise_filename in
                                                 augmentation_noise_filenames]
    augmentation_segment = {
        'signal_filepath':signal_filepath,
        'augmentation_signal_filepath':augmentation_signal_filepath,
        'augmentation_noise_filepaths':augmentation_noise_filepaths,
        'labels':labels
    }

    return augmentation_segment

def read_file2labels(file2labels_filepath):
    labels = {}
    with open(file2labels_filepath) as csvfile:
        file2labels = csv.reader(csvfile, delimiter=',')
        nb_files = 0
        for row in file2labels:
            nb_files+=1
            if len(row) > 1:
                labels[row[0]] = row[1:]
            else:
                labels[row[0]] = []
    return labels


def load_data(data_filepath=None, file2labels_filepath=None, size=300,
              nb_classes=19, image_shape=(1247, 257)):
    if not os.path.isdir(data_filepath):
        raise ValueError("data filepath is invalid")
    if not os.path.isfile(file2labels_filepath):
        nb_csvfiles = 0
        # TODO: May be a performance issue with large datasets!
        for f in os.listdir(data_filepath):
            if f.endswith(".csv"):
                nb_csvfiles+=1
                file2labels_filepath=f
        if nb_csvfiles > 1:
            warnings.warn("There are multiple .csv files in dir: " + data_filepath)

    labels = read_file2labels(file2labels_filepath)

    batch = []
    for i in range(size):
        # TODO: should probably be without replacement
        rand_data_file = random.choice(glob.glob(os.path.join(data_filepath,
                                                              "*.wav.gz")))
        basename = os.path.basename(rand_data_file)
        basenameWithoutExtension = os.path.splitext(basename)[0]
        rand_data_file_labels = labels[basenameWithoutExtension]
        batch.append({'file_name':rand_data_file,
                      'labels':rand_data_file_labels})

    spec_cols, spec_rows = image_shape
    X_train = np.array([]).reshape(0, spec_rows, spec_cols)
    Y_train = np.array([]).reshape(0, nb_classes)
    for sample in batch:
        fs, wave = utils.read_gzip_wave_file(sample['file_name'])
        (f, t, Sxx) = utils.wave_to_spectrogram(wave=wave, fs=fs)
        y = [int(x) for x in sample['labels']]
        y = id_labels2binary_labels(y, nb_classes)
        X_train = np.concatenate((X_train, np.array([Sxx])), axis=0)
        Y_train = np.concatenate((Y_train, np.array([y])), axis=0)

    X_train = X_train.reshape(X_train.shape[0], spec_rows, spec_cols, 1)
    return X_train, Y_train

def load_all_data(data_filepath=None, file2labels_filepath=None, nb_classes=10,
                 image_shape=(32, 32)):
    if not os.path.isdir(data_filepath):
        raise ValueError("data filepath is invalid")
    if not os.path.isfile(file2labels_filepath):
        nb_csvfiles = 0
        # TODO: May be a performance issue with large datasets!
        for f in os.listdir(data_filepath):
            if f.endswith(".csv"):
                nb_csvfiles+=1
                file2labels_filepath=f
        if nb_csvfiles > 1:
            warnings.warn("There are multiple .csv files in dir: " + data_filepath)

    labels = read_file2labels(file2labels_filepath)
    batch = []

    all_data_files = glob.glob(os.path.join(data_filepath, "*.wav.gz"))
    for data_file in all_data_files:
        basenameWithoutExtension = utils.get_basename_without_ext(data_file)
        data_file_labels = labels[basenameWithoutExtension]
        batch.append({'file_name':data_file,
                      'labels':data_file_labels})

    spec_cols, spec_rows = image_shape
    #X_train = np.array([]).reshape(0, spec_rows, spec_cols)
    #Y_train = np.array([]).reshape(0, nb_classes)
    X_train = []
    Y_train = []
    for sample in batch:
        fs, wave = utils.read_gzip_wave_file(sample['file_name'])
        (f, t, Sxx) = utils.wave_to_spectrogram(wave=wave, fs=fs)
        y = [int(x) for x in sample['labels']]
        y = id_labels2binary_labels(y, nb_classes)
        #X_train = np.concatenate((X_train, np.array([Sxx])), axis=0)
        #Y_train = np.concatenate((Y_train, np.array([y])), axis=0)
        X_train.append(Sxx)
        Y_train.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                              X_train.shape[2], 1)
    return X_train, Y_train

def id_labels2binary_labels(labels, nb_classes):
    binary_labels = np.zeros(nb_classes)
    for l in labels:
        binary_labels[l] = 1

    return np.array(binary_labels)

