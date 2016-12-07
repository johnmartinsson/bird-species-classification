from matplotlib import pyplot as plt

import glob
import tqdm

from bird import utils

def compute_and_save_spectrograms_for_files(files):
    progress = tqdm.tqdm(range(len(files)))
    for (f, p) in zip(files, progress):
        img_log_spectrogram_from_wave_file(f)
        img_spectrogram_from_wave_file(f)

def img_log_spectrogram_from_wave_file(filepath):
    fs, x = utils.read_gzip_wave_file(filepath)
    Sxx = utils.wave_to_log_spectrogram_aux(x, fs)
    baseName = utils.get_basename_without_ext(filepath)
    save_matrix_to_file(Sxx, "Log Amplitude Spectrogram", baseName + "_LOG.png")

def img_spectrogram_from_wave_file(filepath):
    fs, x = utils.read_gzip_wave_file(filepath)
    Sxx = utils.wave_to_spectrogram_aux(x, fs)
    baseName = utils.get_basename_without_ext(filepath)
    save_matrix_to_file(Sxx, "Amplitude Spectrogram", baseName + "_AMP.png")

def save_matrix_to_file(Sxx, title, filename):
    cmap = plt.cm.get_cmap('jet')
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Window Samples')
    fig.savefig(filename)
    # close
    plt.clf()
    plt.close()

def plot_log_spectrogram_from_wave_file(filename):
    fs, x = utils.read_gzip_wave_file(filename)
    Sxx = utils.wave_to_log_spectrogram_aux(x, fs)
    plot_matrix(Sxx, "Log Amplitude Spectrogram")

def plot_spectrogram_from_wave_file(filename):
    fs, x = utils.read_gzip_wave_file(filename)
    Sxx = utils.wave_to_spectrogram_aux(x, fs)
    plot_matrix(Sxx, "Amplitude Spectrogram")

def subplot_image(Sxx, n_subplot, title):
    #cmap = grayify_cmap('cubehelix_r')
    cmap = grayify_cmap('jet')
    plt.subplot(n_subplot)
    plt.title(title)
    plt.pcolormesh(Sxx, cmap=cmap)

def plot_matrix(Sxx, title):
    #cmap = grayify_cmap('cubehelix_r')
    cmap = plt.cm.get_cmap('jet')
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Window Samples')
    plt.show()

def plot_vector(x):
    mesh = np.zeros((257, x.shape[0]))
    for i in range(x.shape[0]):
        mesh[256][i] = x[i] * 2500
        mesh[255][i] = x[i] * 2500
        mesh[254][i] = x[i] * 2500
    plot_matrix(mesh)

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
