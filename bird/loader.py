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

def mini_batch_generator(nb_augmentation_samples, nb_mini_batches, batch_size,
                         data_filepath, file2labels_filepath, nb_classes,
                         samplerate):
    """ Create a mini-batch generator
    """
    i_batch_counter = 0
    augmentation_set = create_augmentation_set(data_filepath,
                                               file2labels_filepath,
                                               nb_augmentation_samples)
    while i_batch_counter < nb_mini_batches:
        mini_batch = np.random.choice(augmentation_set, batch_size)

        X_train, Y_train = mini_batch_to_training_data(mini_batch, samplerate,
                                                       nb_classes)

        yield  X_train, Y_train
        i_batch_counter += 1

def augmented_batch_generator(data_filepath, file2labels_filepath, samplerate, nb_mini_batches, nb_classes):

    i_batch_counter = 0
    while i_batch_counter < nb_mini_batches:
        augmentation_set = create_augmentation_set(data_filepath,
                                                   file2labels_filepath,
                                                   nb_classes)
        mini_batch = augmentation_set
        X_train, Y_train = mini_batch_to_training_data(mini_batch, samplerate, nb_classes)

        yield  X_train, Y_train
        i_batch_counter += 1

def mini_batch_to_training_data(mini_batch, samplerate, nb_classes):
    print("load mini-batch...")
    signals_and_labels = [da.apply_augmentation(d, time_shift=False) for d in mini_batch]
    # prepare training samples
    prepare_tmp = [prepare_training_sample(s, l, samplerate, nb_classes) for (s, l) in signals_and_labels]
    X_train = np.concatenate([ss for (ss, ls) in prepare_tmp])
    Y_train = np.concatenate([ls for (ss, ls) in prepare_tmp])
    # force the garbage collector to clear unreferenced memory
    gc.collect()
    Y_train = np.array(Y_train)
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                                        X_train.shape[2], 1)
    return  X_train, Y_train


def prepare_validation_sample(signal, labels, samplerate, nb_classes):
    return prepare_training_sample(signal, labels, samplerate, nb_classes, shift=False)

def prepare_training_sample(signal, labels, samplerate, nb_classes, shift=True):
    """ Prepare training sample by conversion into the spectral domain, and by
    splitting it into equal segments. The last segment is zero padded to the
    desired size.
    """

    # convert signal into the spectral domain
    Sxx = sp.wave_to_sample_spectrogram(signal, samplerate)

    # TODO: I could time/pitch shift the samples here instead
    if shift:
        Sxx = da.time_shift_spectrogram(Sxx)
        Sxx = da.pitch_shift_spectrogram(Sxx)

    # split the spectrogram into equal length segments
    segments = split_into_segments(Sxx, 512)

    # convert the categorical labels into binary labels
    binary_labels = id_labels2binary_labels(labels, nb_classes)

    # replicate labels for each signal segment
    segment_labels = np.array([binary_labels,]*len(segments))

    return segments, segment_labels


def split_into_segments(spectrogram, segment_size):
    """ Split a spectrogram into segments of segment_size. Zero pad the last
    segment.
    """
    spectrogram_size = spectrogram.shape[1]
    nb_padding = segment_size - (spectrogram_size % segment_size)
    padded_spectrogram = np.lib.pad(spectrogram, ((0, 0), (0, nb_padding)), 'constant',
                               constant_values=(0, 0))
    nb_segments = padded_spectrogram.shape[1]/segment_size
    segments = np.split(padded_spectrogram, nb_segments, axis=1)
    return segments

def load_noise_segment_filenames(data_filepath):
    """ Load the filepaths to the noise segments
    """
    noise_segment_filenames = [utils.get_basename_without_ext(f) for f in
                           glob.glob(os.path.join(data_filepath, "noise",
                                                  "*.wav.gz"))]
    return noise_segment_filenames

def load_signal_segment_filenames(data_filepath):
    """ Load the filepaths to the signal segments
    """
    signal_segment_filenames = [utils.get_basename_without_ext(f) for f in
                            glob.glob(os.path.join(data_filepath, "*.wav.gz"))]
    return signal_segment_filenames

def create_augmentation_set_new(data_filepath, file2labels_filepath):
    """ Augment each training sample once """
    signal_segment_filenames = load_signal_segment_filenames(data_filepath)
    noise_segment_filenames = load_noise_segment_filenames(data_filepath)
    file2labels = read_file2labels(file2labels_filepath)
    augmentation_set = []

    for signal_filename in signal_segment_filenames:
        augmentation_dict = da.create_augmentation_dict(signal_filename,
                                                        noise_segment_filenames,
                                                        file2labels)
        augmentation_dict = reconstruct_relative_paths(augmentation_dict,
                                                       data_filepath)
        augmentation_set.append(augmentation_dict)

    return augmentation_set

def create_augmentation_set(data_filepath, file2labels_filepath, nb_augmentation_dicts):
    """ Create an augmentation set with the specified number of augmentation dicts
    """
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
    """ Reconstruct the relative filepaths from the basename of the files in the
    augmentation dict. Assumes that all noise files are in the folder
    <data_filepath>/noise.
    """
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
    """ Read a file2labels.csv file, and return a dict which maps basenames of
    files to their corresponding labels.
    """
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

def load_test_data(data_filepath, file2labels_filepath, nb_classes):
    if not os.path.isdir(data_filepath):
        raise ValueError("data filepath is invalid")
    if not os.path.isfile(file2labels_filepath):
        raise ValueError("file2labels filepath is not valid")

    labels = read_file2labels(file2labels_filepath)
    all_data_files = glob.glob(os.path.join(data_filepath, "*.wav.gz"))

    X_test = []
    Y_test = []
    for data_file in all_data_files:
        basename = utils.get_basename_without_ext(data_file)
        data_file_labels = labels[basename]

        fs, wave = utils.read_gzip_wave_file(data_file)
        categorical_labels = [int(x) for x in data_file_labels]

        signal_segments, signal_segment_labels = prepare_validation_sample(wave, categorical_labels, fs, nb_classes)
        X_test.append(np.asarray(signal_segments))
        Y_test.append(np.asarray(signal_segment_labels))

    return X_test, Y_test

def load_validation_data(data_filepath=None, file2labels_filepath=None, nb_classes=10):
    """ Load the validation data
    """
    if not os.path.isdir(data_filepath):
        raise ValueError("data filepath is invalid")
    if not os.path.isfile(file2labels_filepath):
        raise ValueError("file2labels filepath is not valid")

    labels = read_file2labels(file2labels_filepath)
    batch = []

    all_data_files = glob.glob(os.path.join(data_filepath, "*.wav.gz"))
    for data_file in all_data_files:
        basenameWithoutExtension = utils.get_basename_without_ext(data_file)
        data_file_labels = labels[basenameWithoutExtension]
        batch.append({'file_name':data_file,
                      'labels':data_file_labels})

    X_valid = []
    Y_valid = []
    for sample in batch:
        fs, wave = utils.read_gzip_wave_file(sample['file_name'])
        y = [int(x) for x in sample['labels']]
        signal_segments, signal_segment_labels = prepare_validation_sample(wave, y, fs, nb_classes)
        X_valid.append(signal_segments)
        Y_valid.append(signal_segment_labels)

    X_valid = np.concatenate(X_valid)
    Y_valid = np.concatenate(Y_valid)

    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)

    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1],
                              X_valid.shape[2], 1)
    return X_valid, Y_valid

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

