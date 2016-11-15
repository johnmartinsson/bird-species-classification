# Given path to .wav folder, and path to .csv file name -> class bag
# Return training data X_train, Y_train

import csv
import os
import warnings
import random
import glob
import utils
import numpy as np

def load_data(data_filepath=None, file2labels_filepath=None, size=300,
              nb_classes=19, image_shape=(257, 624)):
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

    labels = {}
    with open(file2labels_filepath, newline='') as csvfile:
        file2labels = csv.reader(csvfile, delimiter=',')
        nb_files = 0
        for row in file2labels:
            nb_files+=1
            if len(row) > 1:
                labels[row[0]] = row[1:]
            else:
                labels[row[0]] = []
        #print("Number of files: ", nb_files)

    batch = []
    for i in range(size):
        # TODO: should probably be without replacement
        rand_data_file = random.choice(glob.glob(os.path.join(data_filepath, "*.wav")))
        basename = os.path.basename(rand_data_file)
        basenameWithoutExtension = os.path.splitext(basename)[0]
        rand_data_file_labels = labels[basenameWithoutExtension]
        batch.append({'file_name':rand_data_file,
                      'labels':rand_data_file_labels})

    spec_rows, spec_cols = image_shape
    X_train = np.array([]).reshape(0, spec_rows, spec_cols)
    Y_train = np.array([]).reshape(0, nb_classes)
    for sample in batch:
        fs, wave = utils.read_wave_file(sample['file_name'])
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

    labels = {}
    with open(file2labels_filepath, newline='') as csvfile:
        file2labels = csv.reader(csvfile, delimiter=',')
        nb_files = 0
        for row in file2labels:
            nb_files+=1
            if len(row) > 1:
                labels[row[0]] = row[1:]
            else:
                labels[row[0]] = []

    batch = []

    all_data_files = glob.glob(os.path.join(data_filepath, "*.wav"))
    for data_file in all_data_files:
        # TODO: should probably be without replacement
        basename = os.path.basename(data_file)
        basenameWithoutExtension = os.path.splitext(basename)[0]
        data_file_labels = labels[basenameWithoutExtension]
        batch.append({'file_name':data_file,
                      'labels':data_file_labels})

    spec_rows, spec_cols = image_shape
    X_train = np.array([]).reshape(0, spec_rows, spec_cols)
    Y_train = np.array([]).reshape(0, nb_classes)
    for sample in batch:
        fs, wave = utils.read_wave_file(sample['file_name'])
        (f, t, Sxx) = utils.wave_to_spectrogram(wave=wave, fs=fs)
        y = [int(x) for x in sample['labels']]
        y = id_labels2binary_labels(y, nb_classes)
        X_train = np.concatenate((X_train, np.array([Sxx])), axis=0)
        Y_train = np.concatenate((Y_train, np.array([y])), axis=0)

    X_train = X_train.reshape(X_train.shape[0], spec_rows, spec_cols, 1)
    return X_train, Y_train

def id_labels2binary_labels(labels, nb_classes):
    binary_labels = np.zeros(nb_classes)
    for l in labels:
        binary_labels[l] = 1

    return np.array(binary_labels)

