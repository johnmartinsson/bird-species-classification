import os
import glob
import numpy as np
# Fix the random seed to make results reproducible
np.random.seed(42)

from bird import utils
from functools import reduce

def time_shift(spectrogram):
    return

def pitch_shift(spectrogram):
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

def apply_augmentation(augmentation_dict):
    """ Load the wave segments from file and apply the augmentation
    """
    fs, s1 = utils.read_wave_file(augmentation_dict['signal_filepath'])
    fs, s2 = utils.read_wave_file(augmentation_dict['augmentation_signal_filepath'])
    noise_segments_aux = map(utils.read_wave_file, augmentation_dict['augmentation_noise_filepaths'])
    noise_segments = [n for (fs, n) in noise_segments_aux]
    augmentation_segments = [s1, s2] + noise_segments
    s_aug = reduce(lambda s1, s2: additively_combine_narrays(s1, s2), augmentation_segments)

    labels = augmentation_dict['labels']
    labels = [int(l) for l in labels]
    # return augmented signal and its labels
    return (s_aug, labels)

def additively_combine_narrays(narr1, narr2):
    """ Additively combine two narrays using a random weight
    # Arguments
        narr1    : a narray
        narr2    : a narray
    # Returns
        out     : the combined and rescaled narray
    """
    alpha = np.random.rand()
    combined_narray = alpha*narr1 + (1.0-alpha)*narr2
    return combined_narray

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
