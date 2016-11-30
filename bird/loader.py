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
    """ Create a mini-batch generator
    """
    i_batch_counter = 0
    augmentation_set = create_augmentation_set(data_filepath,
                                               file2labels_filepath,
                                               nb_augmentation_samples)
    while i_batch_counter < nb_mini_baches:
        mini_batch = np.random.choice(augmentation_set, batch_size)
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
        yield  X_train, Y_train
        i_batch_counter += 1

def remove_low_and_high_frequency_bins(spectrogram, nb_low=5, nb_high=25):
    """ Removes low and high frequency bins from a spectrogram

    # Argmuents
        nb_low  : the number of lower frequency bins to remove
        nb_high : the number of higher frequency bins to remove
    # Returns
        spectrogram : the narrowed spectrogram
    """
    return spectrogram[nb_low:spectrogram.shape[0]-nb_high]

def prepare_validation_sample(signal, labels, samplerate, nb_classes):
    return prepare_training_sample(signal, labels, samplerate, nb_classes, shift=False)

def prepare_training_sample(signal, labels, samplerate, nb_classes, shift=True):
    """ Prepare training sample by conversion into the spectral domain, and by
    splitting it into equal segments. The last segment is zero padded to the
    desired size.
    """

    # convert signal into the spectral domain
    Sxx = utils.wave_to_log_spectrogram_aux(signal, samplerate)

    # TODO: I could time/pitch shift the samples here instead
    if shift:
        Sxx = da.time_shift_spectrogram(Sxx)
        Sxx = da.pitch_shift_spectrogram(Sxx)

    # remove lowest and highest frequency bins
    #Sxx = remove_low_and_high_frequency_bins(Sxx, nb_low=4, nb_high=25)

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

def load_validation_data(data_filepath=None, file2labels_filepath=None, nb_classes=10,
                 image_shape=(32, 32)):
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

    X_train = []
    Y_train = []
    for sample in batch:
        fs, wave = utils.read_gzip_wave_file(sample['file_name'])
        y = [int(x) for x in sample['labels']]
        signal_segments, signal_segment_labels = prepare_validation_sample(wave, y, fs, nb_classes)
        X_train.append(signal_segments)
        Y_train.append(signal_segment_labels)

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                              X_train.shape[2], 1)
    return X_train, Y_train

def id_labels2binary_labels(labels, nb_classes):
    """ Convert categorical labels to binary labels
    """
    binary_labels = np.zeros(nb_classes)
    for l in labels:
        binary_labels[l] = 1

    return np.array(binary_labels)

