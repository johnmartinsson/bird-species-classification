from matplotlib import pyplot as plt

import pickle
import glob
import tqdm
import numpy as np
from skimage import morphology

from bird import utils
from bird import preprocessing as pp
from bird import signal_processing as sp

def compute_and_save_spectrograms_for_files(files):
    progress = tqdm.tqdm(range(len(files)))
    for (f, p) in zip(files, progress):
        img_log_spectrogram_from_wave_file(f)
        img_spectrogram_from_wave_file(f)

def img_log_spectrogram_from_wave_file(filepath):
    fs, x = utils.read_gzip_wave_file(filepath)
    Sxx = sp.wave_to_amplitude_spectrogram(x, fs, 512, 128)[:256]
    baseName = utils.get_basename_without_ext(filepath)
    save_matrix_to_file(Sxx, "Amplitude Spectrogram", baseName + "_LOG.png")

def img_spectrogram_from_wave_file(filepath):
    fs, x = utils.read_gzip_wave_file(filepath)
    Sxx = sp.wave_to_log_amplitude_spectrogram(x, fs, 512, 128)[:256]
    baseName = utils.get_basename_without_ext(filepath)
    save_matrix_to_file(Sxx, "Log Amplitude Spectrogram", baseName + "_AMP.png")

def sprengel_binary_mask_from_wave_file(filepath):
    fs, x = utils.read_gzip_wave_file(filepath)
    Sxx = sp.wave_to_amplitude_spectrogram(x, fs, 512, 128)[256:]

    # plot spectrogram
    plt.figure(1)
    subplot_image(Sxx, 411, "Spectrogram")

    Sxx = pp.normalize(Sxx)
    binary_image = pp.median_clipping(Sxx, 3.0)

    subplot_image(binary_image + 0, 412, "Median Clipping")

    binary_image = morphology.binary_erosion(binary_image, selem=np.ones((4, 4)))

    subplot_image(binary_image + 0, 413, "Erosion")

    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    subplot_image(binary_image + 0, 414, "Dilation")
    fig = plt.figure(1)

    mask = np.array([np.max(col) for col in binary_image.T])
    mask = morphology.binary_dilation(mask, np.ones(4))
    mask = morphology.binary_dilation(mask, np.ones(4))

    plot_vector(mask, "Mask")

    fig.set_size_inches(10, 12)
    plt.tight_layout()
    fig.savefig(utils.get_basename_without_ext(filepath) + "_binary_mask.png", dpi=100)

def signal_and_noise_spectrogram_from_wave_file(filepath):

    (fs, wave) = utils.read_wave_file(filepath)
    spectrogram = sp.wave_to_amplitude_spectrogram(wave, fs)
    signal_wave, noise_wave = pp.preprocess_wave(wave, fs)
    spectrogram_signal = sp.wave_to_log_amplitude_spectrogram(wave, fs)
    spectrogram_noise = sp.wave_to_log_amplitude_spectrogram(wave, fs)

    fig = plt.figure(1)
    cmap = plt.cm.get_cmap('jet')
    # whole spectrogram
    plt.subplot(121)
    plt.pcolormesh(spectrogram, cmap=cmap)
    plt.title("Whole Sound")

    plt.subplot(122)
    plt.pcolormesh(spectrogram_signal, cmap=cmap)
    plt.title("Signal Part")

    plt.subplot(123)
    plt.pcolormesh(spectrogram_noise, cmap=cmap)
    plt.title("Noise Part")

    basename = utils.get_basename_without_ext(filepath)
    fig.savefig(basename+"_noise_signal.png")

    # close
    plt.clf()
    plt.close()

def save_matrix_to_file(Sxx, title, filename):
    cmap = plt.cm.get_cmap('jet')
    #cmap = grayify_cmap('cubehelix_r')
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Window Samples')
    fig.savefig(filename)
    # close
    plt.clf()
    plt.close()

def plot_history_to_image_file(pickle_path):
    with open(pickle_path, 'rb') as input:
        trainLoss = pickle.load(input)
        validLoss = pickle.load(input)
        trainAcc = pickle.load(input)
        validAcc = pickle.load(input)
        plt.figure(1)
        plt.subplot(211)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.plot(trainLoss, 'v', label="train")
        plt.plot(validLoss, 'o', label="valid")
        plt.legend(loc="upper_left")
        plt.subplot(212)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.plot(trainAcc, 'v', label="train")
        plt.plot(validAcc, 'o', label="valid")
        plt.legend(loc="upper_left")
        plt.show()

def plot_log_spectrogram_from_wave_file(filename):
    fs, x = utils.read_gzip_wave_file(filename)
    Sxx = sp.wave_to_log_amplitude_spectrogram(x, fs, 512, 128)[:256]
    Sxx2 = utils.wave_to_log_spectrogram_aux(x, fs)
    plot_matrix(Sxx, "Log Amplitude Spectrogram")
    plot_matrix(Sxx2, "Log Amplitude Spectrogram (scipy)")

def plot_spectrogram_from_wave_file(filename):
    fs, x = utils.read_gzip_wave_file(filename)
    Sxx = sp.wave_to_amplitude_spectrogram(x, fs, 512, 128)[:256]
    plot_matrix(Sxx, "Amplitude Spectrogram")

def subplot_image(Sxx, n_subplot, title):
    cmap = grayify_cmap('cubehelix_r')
    # cmap = grayify_cmap('jet')
    plt.subplot(n_subplot)
    plt.title(title)
    plt.pcolormesh(Sxx, cmap=cmap)

def plot_matrix(Sxx, title):
    # cmap = grayify_cmap('cubehelix_r')
    cmap = plt.cm.get_cmap('jet')
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Window Samples')
    plt.show()

def plot_vector(x, title):
    mesh = np.zeros((257, x.shape[0]))
    for i in range(x.shape[0]):
        mesh[256][i] = x[i] * 2500
        mesh[255][i] = x[i] * 2500
        mesh[254][i] = x[i] * 2500
    plot_matrix(mesh, title)

def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)
