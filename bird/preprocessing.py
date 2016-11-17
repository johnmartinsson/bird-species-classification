import numpy as np
import utils
from skimage import morphology

def compute_noise_mask(spectrogram):
    norm_spectrogram = normalize(spectrogram)
    binary_structure = mark_cells_times_larger_than_median(norm_spectrogram, 2.5)

    n_hood = np.ones((4, 4))
    binary_structure = morphology.binary_erosion(binary_structure, n_hood)
    binary_structure = morphology.binary_dilation(binary_structure, n_hood)

    utils.plot_matrix(binary_structure)

    indicator_vector = np.array([set_one_if_column_contains_one(col) for col in np.transpose(binary_structure)])
    indicator_vector = smooth_indicator_vector(indicator_vector)

    # flip 0s with 1s and vice versa
    return 1-indicator_vector

def compute_structure_mask(spectrogram):
    norm_spectrogram = normalize(spectrogram)
    binary_structure = mark_cells_times_larger_than_median(norm_spectrogram, 3)

    n_hood = np.ones((4, 4))
    binary_structure = morphology.binary_dilation(binary_structure, n_hood)

    # plot it
    utils.plot_matrix(binary_structure)

    binary_structure = morphology.binary_erosion(binary_structure, n_hood)

    # plot it
    #utils.plot_matrix(binary_structure)

    indicator_vector = np.array([set_one_if_column_contains_one(col) for col in np.transpose(binary_structure)])
    indicator_vector = smooth_indicator_vector(indicator_vector)

    return indicator_vector

def smooth_indicator_vector(indicator_vector):
    n_hood = np.ones(4)
    indicator_vector = morphology.binary_dilation(indicator_vector, n_hood)
    indicator_vector = morphology.binary_dilation(indicator_vector, n_hood)

    # type casting is a bitch
    return 1*indicator_vector

def set_one_if_column_contains_one(col):
    nb_ones = np.count_nonzero(col)
    if nb_ones > 0:
        return int(1)
    else:
        return int(0)

def mark_cells_times_larger_than_median(spectrogram, number_times_larger):
    row_medians = np.median(spectrogram, axis=1)
    col_medians = np.median(spectrogram, axis=0)
    for row in range(spectrogram.shape[0]):
        for col in range(spectrogram.shape[1]):
            if spectrogram[row][col] > 3*row_medians[row] \
               and spectrogram[row][col] > 3*col_medians[col]:
                spectrogram[row][col] = 1
            else:
                spectrogram[row][col] = 0

    return spectrogram


def extract_signal(mask, spectrogram):
    print("extract_signal is undefined")


def normalize(X):
    mi = np.min(X)
    ma = np.max(X)

    X = (X-mi)/(ma-mi)
    return X
