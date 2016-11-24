import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import skimage.filters as filters
import glob
import os
import csv
import tqdm

from bird import utils
from bird import loader

def preprocess_data_set(data_path, output_directory):
    wave_files = glob.glob(os.path.join(data_path, "*.wav"))
    file2labels_path = os.path.join(data_path, "file2labels.csv")

    file2labels = loader.read_file2labels(file2labels_path);
    file2labels_path_new = os.path.join(output_directory, "file2labels.csv")

    with open(file2labels_path_new, 'w') as file2labels_csv:
        file2labelswriter = csv.writer(file2labels_csv)

        progress = tqdm.tqdm(range(len(wave_files)))
        for (f, p) in zip(wave_files, progress):
            basename = utils.get_basename_without_ext(f)
            labels = file2labels[basename]
            preprocess_sound_file(f, output_directory, labels,
                                  file2labelswriter)

def preprocess_wave(wave, fs):
    (t, f, Sxx) = utils.wave_to_spectrogram(wave, fs)

    n_mask = compute_noise_mask(Sxx)
    s_mask = compute_signal_mask(Sxx)

    n_mask_scaled = reshape_binary_mask(n_mask, wave.shape[0])
    s_mask_scaled = reshape_binary_mask(s_mask, wave.shape[0])

    signal_wave = extract_masked_part_from_wave(s_mask_scaled, wave)
    noise_wave = extract_masked_part_from_wave(n_mask_scaled, wave)

    chunk_size = 512 * 128
    signal_wave_padded = zero_pad_wave(signal_wave, chunk_size)
    noise_wave_padded = zero_pad_wave(noise_wave, chunk_size)

    signal_chunks = split_into_chunks(signal_wave_padded, chunk_size)
    noise_chunks = split_into_chunks(noise_wave_padded, chunk_size)

    return signal_chunks, noise_chunks

def preprocess_sound_file(filename, output_directory, labels, file2labelswriter):
    basename = os.path.splitext(os.path.basename(filename))[0]
    fs, x = utils.read_wave_file(filename)
    signal_chunks, noise_chunks = preprocess_wave(x, fs)

    i_chunk = 0
    for s in signal_chunks:
        filename_chunk = os.path.join(output_directory, basename +
                                      "_signal_chunk_" + str(i_chunk) + ".wav")
        utils.write_wave_to_file(filename_chunk, fs, s)
        file2labelswriter.writerow([utils.get_basename_without_ext(filename_chunk)] + labels)
        i_chunk += 1

    i_chunk = 0
    for s in noise_chunks:
        filename_chunk = os.path.join(output_directory, basename +
                                      "_noise_chunk_" + str(i_chunk) + ".wav")
        utils.write_wave_to_file(filename_chunk, fs, s)
        i_chunk += 1

def split_into_chunks(array, chunk_size):
    nb_chunks = array.shape[0]/chunk_size

    return np.split(array, nb_chunks)

def zero_pad_wave(wave, chunk_size):
    nb_wave = wave.shape[0]
    nb_padding = chunk_size - (nb_wave % chunk_size)
    return np.lib.pad(wave, (0, nb_padding), 'constant', constant_values=(0, 0))

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
    mask = compute_binary_mask(spectrogram, threshold)
    return mask

def compute_noise_mask(spectrogram):
    """ Computes a binary noise mask (convenience function)

    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)

    # Returns
        binary_mask : the binary noise mask
    """
    threshold = 2.5
    mask = compute_binary_mask(spectrogram, threshold)
    # invert mask
    return np.logical_not(mask)

def compute_binary_mask(spectrogram, threshold, save_as_image=False, filename=""):
    """ Computes a binary mask for the spectrogram
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
        threshold   : a threshold for times larger than the median
    # Returns
        binary_mask : the binary mask
    """
    norm_spectrogram = normalize(spectrogram)
    binary_image = mark_cells_times_larger_than_median(norm_spectrogram, threshold)

    if save_as_image:
        plt.figure(1)
        utils.subplot_image(spectrogram, 411, "Spectrogram")
        utils.subplot_image(binary_image, 412, "Median Clipping")

    # closing binary image (dilation followed by erosion)
    binary_image = morphology.binary_closing(binary_image, selem=np.ones((4, 4)))

    # dialate binary image
    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    if save_as_image:
        utils.subplot_image(binary_image, 413, "Closing and Dilation")

    # apply median filter
    #binary_image = filters.median(binary_image, selem=np.ones((2, 2)))

    # remove small objects
    binary_image = morphology.remove_small_objects(binary_image, min_size=32,
                                                   connectivity=1)
    if save_as_image:
        utils.subplot_image(binary_image, 414, "Small Objects Removed")

    if save_as_image:
        fig = plt.figure(1)
        fig.set_size_inches(10, 12)
        plt.tight_layout()
        fig.savefig(filename, dpi=100)

    # TODO: transpose is O(n^2)
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

    #print("scale_fact", scale_fact)
    #print("i_end:", i_end)
    #print("size:", size)

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

def mark_cells_times_larger_than_median(spectrogram, number_times_larger):
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

def remove_low_and_high_frequency_bins(spectrogram, nb_low=5, nb_high=25):
    """ Removes low and high frequency bins from a spectrogram

    # Argmuents
        nb_low  : the number of lower frequency bins to remove
        nb_high : the number of higher frequency bins to remove
    # Returns
        spectrogram : the narrowed spectrogram
    """
    return spectrogram[nb_low:spectrogram.shape[0]-nb_high]

def normalize(X):
    """ Normalize numpy array to interval [0, 1]
    """
    mi = np.min(X)
    ma = np.max(X)

    X = (X-mi)/(ma-mi)
    return X
