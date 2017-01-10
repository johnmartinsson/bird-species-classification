import numpy as np
# Fix the random seed to make results reproducible
np.random.seed(42)

import os
import glob

from bird import utils
from functools import reduce

def time_shift_signal(wave):
    """ Shift a wave in the time-domain at random
    """
    size = wave.shape[0]
    shift_at_time = np.random.randint(0, size)
    return np.roll(wave, shift_at_time)

def pitch_shift_signal(spectrogram):
    return

def time_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the time axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)

def pitch_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols//20 # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)

def same_class_augmentation(wave, class_dir):
    """ Perform same class augmentation of the wave by loading a random segment
    from the class_dir and additively combine the wave with that segment.
    """
    sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    aug_sig_path = np.random.choice(sig_paths, 1, replace=False)[0]
    (fs, aug_sig) = utils.read_wave_file(aug_sig_path)
    alpha = np.random.rand()
    wave = (1.0-alpha)*wave + alpha*aug_sig
    return wave

def noise_augmentation(wave, noise_files):
    """ Perform noise augmentation of the wave by loading three noise segments
    from the noise_dir and add these on top of the wave with a dampening factor
    of 0.4
    """
    aug_noise_files = np.random.choice(noise_files, 3, replace=False)
    dampening_factor = 0.4
    for aug_noise_path in aug_noise_files:
        (fs, aug_noise) = utils.read_wave_file(aug_noise_path)
        wave = wave + aug_noise*dampening_factor
    return wave

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
    """ Fit an array to a specified size by either repeating the array, or
    taking the size first elements from the array
    """
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
    """ Load the wave from file and apply the augmentation
    """
    noise_dampening_factor = 0.4
    alpha = np.random.rand()

    # load the signals
    fs, s1 = utils.read_gzip_wave_file(augmentation_dict['signal_filepath'])
    fs, s2 = utils.read_gzip_wave_file(augmentation_dict['augmentation_signal_filepath'])

    # get max signal length
    ma_s = max(s1.shape[0], s2.shape[0])

    # load the noise
    noise_segments_aux = map(utils.read_gzip_wave_file, augmentation_dict['augmentation_noise_filepaths'])
    noise_segments = [n*noise_dampening_factor for (fs, n) in noise_segments_aux]

    augmentation_segments = [alpha*s1, (1.0-alpha)*s2] + noise_segments
    # fit them to the size of the largest signal by cycle until equal length
    augmentation_segments = [fit_to_size(s, ma_s) for s in augmentation_segments]
    # additively combine them
    s_aug = reduce(lambda s1, s2: s1 + s2, augmentation_segments)

    # time shift signal
    if time_shift:
        s_aug = time_shift_signal(s_aug)

    # load the labels
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

    # create the dict
    augmentation_dict = {
        'signal_filename':signal_filename,
        'labels':signal_labels,
        'augmentation_signal_filename':augmentation_signal_filename,
        'augmentation_noise_filenames':augmentation_noise_filenames
    }
    return augmentation_dict
