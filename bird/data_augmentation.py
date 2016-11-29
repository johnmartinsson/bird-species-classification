import numpy as np
# Fix the random seed to make results reproducible
np.random.seed(42)

import os
import glob

from bird import utils
from functools import reduce

def time_shift_signal(wave):
    size = wave.shape[0]
    shift_at_time = np.random.randint(0, size)
    return np.roll(wave, shift_at_time)

def pitch_shift_signal(spectrogram):
    return

def find_same_labels_filepaths(file2labels, labels):
    """ Finds the audio segments which has the same labels as the labels
    supplied
    # Arguments
        file2labels : a dictionary from filename to labels
        labels      : the labels for which want to find same class files
    # Returns
        [files]     : a list of the files which have the same labels
    """
    same_class_files = []
    for key, value in file2labels.items():
        if(labels == value):
            same_class_files.append(key)
    return same_class_files

def fit_to_size(x, size):
    x_size = x.shape[0]
    if x_size < size:
        nb_repeats = int(np.ceil(size/x_size))
        x_tmp = np.tile(x, nb_repeats)
        x_new = x_tmp[:size]
        return x_new
    elif x_size > size:
        x_new = x[:size]
        return x_new
    else:
        return x

def apply_augmentation(augmentation_dict, time_shift=True):
    """ Load the wave segments from file and apply the augmentation
    """
    noise_dampening_factor = 0.4
    alpha = np.random.rand()

    fs, s1 = utils.read_gzip_wave_file(augmentation_dict['signal_filepath'])
    fs, s2 = utils.read_gzip_wave_file(augmentation_dict['augmentation_signal_filepath'])

    # get max signal length
    ma_s = max(s1.shape[0], s2.shape[0])

    noise_segments_aux = map(utils.read_gzip_wave_file, augmentation_dict['augmentation_noise_filepaths'])
    noise_segments = [n*noise_dampening_factor for (fs, n) in noise_segments_aux]

    augmentation_segments = [alpha*s1, (1.0-alpha)*s2] + noise_segments
    # fit all of them to the size of the largest signal by cycle until equal
    # length
    augmentation_segments = [fit_to_size(s, ma_s) for s in augmentation_segments]
    s_aug = reduce(lambda s1, s2: s1 + s2, augmentation_segments)

    # time shift signal
    if time_shift:
        s_aug = time_shift_signal(s_aug)

    labels = augmentation_dict['labels']
    labels = [int(l) for l in labels]
    # return augmented signal and its labels
    return (s_aug, labels)

def create_augmentation_dict(signal_filename, noise_segment_filenames, file2labels):
    """ Create a dict with the paths to the signal segments that should be used
    to create this unique, augmented, sample. Assumes that noise segments is in
    data_path/noise.

    # Arguments
        signal1_filepath : the filepath to the signal segment
        signal1_labels   : the labels of the signal
        file2labels      : a dict from signal filepath to labels
        data_path        : the path to the data sound files

    # Returns
        unique_sample_paths_dict : a dict with the filepaths to the signal and
        noise segments which will be used to augment the signal
    """
    nb_noise_segments = 3
    nb_same_class_segments = 1

    signal_labels = file2labels[signal_filename]
    same_labels_signal_filenames = find_same_labels_filepaths(file2labels, signal_labels)

    augmentation_signal_filename = np.random.choice(same_labels_signal_filenames,
                                        nb_same_class_segments,
                                        replace=False)[0]

    augmentation_noise_filenames = np.random.choice(noise_segment_filenames, nb_noise_segments, replace=False)

    # reconstruct relative paths
    #signal1_relative_filepath = os.path.join(data_path, signal1_filepath + ".wav")
    #signal2_relative_filepath = os.path.join(data_path, signal2_filepath + ".wav")
    #noise_relative_filepaths = [os.path.join(data_path, n_path) for n_path in noise_filepaths]

    # create the dict
    augmentation_dict = {
        'signal_filename':signal_filename,
        'labels':signal_labels,
        'augmentation_signal_filename':augmentation_signal_filename,
        'augmentation_noise_filenames':augmentation_noise_filenames
    }
    return augmentation_dict
