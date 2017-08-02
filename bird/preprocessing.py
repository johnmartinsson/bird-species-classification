import numpy as np
from skimage import morphology
import skimage.filters as filters
import glob
import os
import csv
import tqdm
import scipy.signal as sps

from bird import utils
from bird import signal_processing as sp

def preprocess_sound_file(filename, class_dir, noise_dir, segment_size_seconds):
    """ Preprocess sound file. Loads sound file from filename, downsampels,
    extracts signal/noise parts from sound file, splits the signal/noise parts
    into equally length segments of size segment size seconds.

    # Arguments
        filename : the sound file to preprocess
        class_dir : the directory to save the extracted signal segments in
        noise_dir : the directory to save the extracted noise segments in
        segment_size_seconds : the size of each segment in seconds
    # Returns
        nothing, simply saves the preprocessed sound segments
    """

    samplerate, wave = utils.read_wave_file_not_normalized(filename)

    if len(wave) == 0:
        print("An empty sound file..")
        wave = np.zeros(samplerate * segment_size_seconds, dtype=np.int16)

    signal_wave, noise_wave = preprocess_wave(wave, samplerate)

    if signal_wave.shape[0] == 0:
        signal_wave = np.zeros(samplerate * segment_size_seconds, dtype=np.int16)

    basename = utils.get_basename_without_ext(filename)

    # print(filename)
    # print(class_dir)
    # print(noise_dir)

    if signal_wave.shape[0] > 0:
        signal_segments = split_into_segments(signal_wave, samplerate, segment_size_seconds)
        save_segments_to_file(class_dir, signal_segments, basename, samplerate)
    if noise_wave.shape[0] > 0:
        noise_segments = split_into_segments(noise_wave, samplerate, segment_size_seconds)
        save_segments_to_file(noise_dir, noise_segments, basename, samplerate)

def save_segments_to_file(output_dir, segments, basename, samplerate):
    # print("save segments ({}) to file".format(str(len(segments))))
    i_segment = 0
    for segment in segments:
        segment_filepath = os.path.join(output_dir, basename + "_seg_" + str(i_segment) + ".wav")
        # print("save segment: {}".format(segment_filepath))
        utils.write_wave_to_file(segment_filepath, samplerate, segment)
        i_segment += 1

def split_into_segments(wave, samplerate, segment_time):
    """ Split a wave into segments of segment_size. Repeat signal to get equal
    length segments.
    """
    # print("split into segments")
    segment_size = samplerate * segment_time
    wave_size = wave.shape[0]

    nb_repeat = segment_size - (wave_size % segment_size)
    nb_tiles = 2
    if wave_size < segment_size:
        nb_tiles = int(np.ceil(segment_size/wave_size))
    repeated_wave = np.tile(wave, nb_tiles)[:wave_size+nb_repeat]
    nb_segments = repeated_wave.shape[0]/segment_size

    if not repeated_wave.shape[0] % segment_size == 0:
        raise ValueError("reapeated wave not even multiple of segment size")

    segments = np.split(repeated_wave, int(nb_segments), axis=0)

    return segments

def preprocess_wave(wave, fs):
    """ Preprocess a signal by computing the noise and signal mask of the
    signal, and extracting each part from the signal
    """
    Sxx = sp.wave_to_amplitude_spectrogram(wave, fs)

    n_mask = compute_noise_mask(Sxx)
    s_mask = compute_signal_mask(Sxx)

    n_mask_scaled = reshape_binary_mask(n_mask, wave.shape[0])
    s_mask_scaled = reshape_binary_mask(s_mask, wave.shape[0])

    signal_wave = extract_masked_part_from_wave(s_mask_scaled, wave)
    noise_wave = extract_masked_part_from_wave(n_mask_scaled, wave)

    return signal_wave, noise_wave

def extract_noise_part(spectrogram):
    """ Extract the noise part of a spectrogram
    """
    mask = compute_noise_mask(spectrogram)
    noise_part = extract_masked_part_from_spectrogram(mask, spectrogram)
    return noise_part

def extract_signal_part(spectrogram):
    """ Extract the signal part of a spectrogram
    """
    mask = compute_signal_mask(spectrogram)
    signal_part = extract_masked_part_from_spectrogram(mask, spectrogram)
    return signal_part

def extract_masked_part_from_spectrogram(mask, spectrogram):
    """ Extract the masked part of the spectrogram
    """
    return spectrogram[:,mask]

def extract_masked_part_from_wave(mask, wave):
    return wave[mask]

def compute_signal_mask(spectrogram):
    """ Computes a binary noise mask (convenience function)

    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)

    # Returns
        binary_mask : the binary signal mask
    """
    threshold = 3
    mask = compute_binary_mask_sprengel(spectrogram, threshold)
    return mask

def compute_noise_mask(spectrogram):
    """ Computes a binary noise mask (convenience function)

    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)

    # Returns
        binary_mask : the binary noise mask
    """
    threshold = 2.5
    mask = compute_binary_mask_sprengel(spectrogram, threshold)
    # invert mask
    return np.logical_not(mask)

def compute_binary_mask_sprengel(spectrogram, threshold):
    """ Computes a binary mask for the spectrogram
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
        threshold   : a threshold for times larger than the median
    # Returns
        binary_mask : the binary mask
    """
    # normalize to [0, 1)
    norm_spectrogram = normalize(spectrogram)

    # median clipping
    binary_image = median_clipping(norm_spectrogram, threshold)

    # erosion
    binary_image = morphology.binary_erosion(binary_image, selem=np.ones((4, 4)))

    # dilation
    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    # extract mask
    mask = np.array([np.max(col) for col in binary_image.T])
    mask = smooth_mask(mask)

    return mask

def compute_binary_mask_lasseck(spectrogram, threshold):
    # normalize to [0, 1)
    norm_spectrogram = normalize(spectrogram)

    # median clipping
    binary_image = median_clipping(norm_spectrogram, threshold)

    # closing binary image (dilation followed by erosion)
    binary_image = morphology.binary_closing(binary_image, selem=np.ones((4, 4)))

    # dialate binary image
    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    # apply median filter
    binary_image = filters.median(binary_image, selem=np.ones((2, 2)))

    # remove small objects
    binary_image = morphology.remove_small_objects(binary_image, min_size=32, connectivity=1)

    mask = np.array([np.max(col) for col in binary_image.T])
    mask = smooth_mask(mask)

    return mask


# TODO: This method needs some real testing
def reshape_binary_mask(mask, size):
    """ Reshape a binary mask to a new larger size
    """
    reshaped_mask = np.zeros(size, dtype=bool)

    x_size_mask = mask.shape[0]
    scale_fact = int(np.floor(size/x_size_mask))
    rest_fact = float(size)/x_size_mask - scale_fact

    rest = rest_fact
    i_begin = 0
    i_end = int(scale_fact + np.floor(rest))
    for i in mask:
        reshaped_mask[i_begin:i_end] = i
        rest += rest_fact
        i_begin = i_end
        i_end = i_end + scale_fact + int(np.floor(rest))
        if rest >= 1:
            rest -= 1.

    if not (i_end - scale_fact) == size:
        raise ValueError("there seems to be a scaling error in reshape_binary_mask")

    return reshaped_mask

def smooth_mask(mask):
    """ Smooths a binary mask using 4x4 dilation
        # Arguments
            mask : the binary mask
        # Returns
            mask : a smoother binary mask
    """
    n_hood = np.ones(4)
    mask = morphology.binary_dilation(mask, n_hood)
    mask = morphology.binary_dilation(mask, n_hood)

    # type casting is a bitch
    return mask

def median_clipping(spectrogram, number_times_larger):
    """ Compute binary image from spectrogram where cells are marked as 1 if
    number_times_larger than the row AND column median, otherwise 0
    """
    row_medians = np.median(spectrogram, axis=1)
    col_medians = np.median(spectrogram, axis=0)

    # create 2-d array where each cell contains row median
    row_medians_cond = np.tile(row_medians, (spectrogram.shape[1], 1)).transpose()
    # create 2-d array where each cell contains column median
    col_medians_cond = np.tile(col_medians, (spectrogram.shape[0], 1))

    # find cells number_times_larger than row and column median
    larger_row_median = spectrogram >= row_medians_cond*number_times_larger
    larger_col_median = spectrogram >= col_medians_cond*number_times_larger

    # create binary image with cells number_times_larger row AND col median
    binary_image = np.logical_and(larger_row_median, larger_col_median)
    return binary_image

def normalize(X):
    """ Normalize numpy array to interval [0, 1]
    """
    mi = np.min(X)
    ma = np.max(X)

    X = (X-mi)/(ma-mi)
    return X
