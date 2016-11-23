import numpy as np
import os
import csv
import glob
import sys
import subprocess
import wave
import tqdm

from scipy import signal
from scipy import fft
from scipy.io import wavfile
from matplotlib import pyplot as plt
from functools import reduce

from bird import preprocessing as pp
from bird import loader as loader

def get_basename_without_ext(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return basename

def preprocess_data_set(data_path, output_directory):
    wave_files = glob.glob(os.path.join(data_path, "*.wav"))
    file2labels_path = os.path.join(data_path, "file2labels.csv")

    file2labels = loader.read_file2labels(file2labels_path);
    file2labels_path_new = os.path.join(output_directory, "file2labels.csv")

    with open(file2labels_path_new, 'w') as file2labels_csv:
        file2labelswriter = csv.writer(file2labels_csv)

        progress = tqdm.tqdm(range(len(wave_files)))
        for (f, p) in zip(wave_files, progress):
            basename = get_basename_without_ext(f)
            labels = file2labels[basename]
            preprocess_sound_file(f, output_directory, labels,
                                  file2labelswriter)

def preprocess_wave(wave, fs):
    (t, f, Sxx) = wave_to_spectrogram(wave, fs)

    n_mask = pp.compute_noise_mask(Sxx)
    s_mask = pp.compute_signal_mask(Sxx)

    n_mask_scaled = pp.reshape_binary_mask(n_mask, wave.shape[0])
    s_mask_scaled = pp.reshape_binary_mask(s_mask, wave.shape[0])

    #print ("Signal shape: ", x.shape)
    #print ("Noise mask shape: ", n_mask_scaled.shape)

    signal_wave = pp.extract_masked_part_from_wave(s_mask_scaled, wave)
    noise_wave = pp.extract_masked_part_from_wave(n_mask_scaled, wave)

    chunk_size = 512 * 128
    signal_wave_padded = zero_pad_wave(signal_wave, chunk_size)
    noise_wave_padded = zero_pad_wave(noise_wave, chunk_size)

    signal_chunks = split_into_chunks(signal_wave_padded, chunk_size)
    noise_chunks = split_into_chunks(noise_wave_padded, chunk_size)

    return signal_chunks, noise_chunks

def preprocess_sound_file(filename, output_directory, labels, file2labelswriter):
    basename = os.path.splitext(os.path.basename(filename))[0]
    fs, x = read_wave_file(filename)
    signal_chunks, noise_chunks = preprocess_wave(x, fs)

    i_chunk = 0
    for s in signal_chunks:
        filename_chunk = os.path.join(output_directory, basename +
                                      "_signal_chunk_" + str(i_chunk) + ".wav")
        write_wave_to_file(filename_chunk, fs, s)
        file2labelswriter.writerow([get_basename_without_ext(filename_chunk)] + labels)
        i_chunk += 1

    i_chunk = 0
    for s in noise_chunks:
        filename_chunk = os.path.join(output_directory, basename +
                                      "_noise_chunk_" + str(i_chunk) + ".wav")
        write_wave_to_file(filename_chunk, fs, s)
        i_chunk += 1

def find_same_class_files(file2labels, labels):
    same_class_files = []
    for key, value in file2labels.items():
        if(labels == value):
            same_class_files.append(key)
    return same_class_files

def split_into_chunks(array, chunk_size):
    nb_chunks = array.shape[0]/chunk_size

    return np.split(array, nb_chunks)

def zero_pad_wave(wave, chunk_size):
    nb_wave = wave.shape[0]
    nb_padding = chunk_size - (nb_wave % chunk_size)
    return np.lib.pad(wave, (0, nb_padding), 'constant', constant_values=(0, 0))


def test(filename):
    fs, x = read_wave_file(filename)
    (t, f, Sxx) = wave_to_spectrogram(x, fs)
    #noise = pp.extract_noise_part(Sxx)
    #plot_matrix(Sxx, "Spectrogram")

    n_mask = pp.compute_noise_mask(Sxx)
    s_mask = pp.compute_signal_mask(Sxx)

    n_mask_scaled = pp.reshape_binary_mask(n_mask, x.shape[0])
    s_mask_scaled = pp.reshape_binary_mask(s_mask, x.shape[0])

    signal_wave = pp.extract_masked_part_from_wave(s_mask_scaled, x)
    noise_wave = pp.extract_masked_part_from_wave(n_mask_scaled, x)

    signal_wave_padded = zero_pad_wave(signal_wave)
    noise_wave_padded = zero_pad_wave(noise_wave)

    (t, f, Sxx_signal) = wave_to_spectrogram(signal_wave, fs)
    (t, f, Sxx_noise) = wave_to_spectrogram(noise_wave, fs)

    #plot_matrix(Sxx_signal, "Signal Spectrogram")
    #plot_matrix(Sxx_noise, "Noise Spectrogram")

def plot_spectrogram_from_wave(filename):
    fs, x = read_wave_file(filename)
    (t, f, Sxx) = wave_to_spectrogram(x, fs)
    plot_matrix(Sxx, "Spectrogram")

def play_wave_file(filename):
    """ Play a wave file
    """
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")
    else:
        if (sys.platform == "linux" or sys.playform == "linux2"):
            subprocess.call(["aplay", filename])
        else:
            print ("Platform not supported")

def write_wave_to_file(filename, rate, wave):
    wavfile.write(filename, rate, wave)

def read_wave_file(filename):
    """ Read a wave file from disk
    # Arguments
        filename : the name of the wave file
    # Returns
        (fs, x)  : (sampling frequency, signal)
    """
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")

    s = wave.open(filename, 'rb')

    if (s.getnchannels() != 1):
        raise ValueError("Wave file should be mono")
    if (s.getframerate() != 16000):
        raise ValueError("Sampling rate of wave file should be 16000")

    strsig = s.readframes(s.getnframes())
    x = np.fromstring(strsig, np.short)
    fs = s.getframerate()
    s.close()

    return fs, x

def wave_to_spectrogram(wave=np.array([]), fs=None, nperseg=512, noverlap=384):
    """Given a wave form returns the spectrogram of the wave form.
    # Arguments
        wave : the wave form (default np.array([]))
        fs   : the rate at which the wave form has been sampled
    # Returns
        spectrogram : the computed spectrogram (numpy array)
    """
    window = signal.get_window('hanning', nperseg)
    return signal.spectrogram(wave, fs, window, nperseg, noverlap,
                              mode='magnitude')

def additively_combine_narrays(narrays):
    """ Additively combine and rescale two narrays
    # Arguments
        narrays : the narrays to combine
    # Returns
        out     : the combined and rescaled narray
    """
    nb_narrays = narrays.shape[0]
    norm_factor = 1.0/nb_narrays
    narray = reduce(lambda a1, a2: norm_factor*a1 + norm_factor*a2, narrays)
    return narray

def wave_to_spectrogram2(S):
    Spectrogram = []
    N = 160000
    K = 512
    Step = 4
    wind =  0.5*(1 -np.cos(np.array(range(K))*2*np.pi/(K-1) ))

    for j in range(int(Step*N/K)-Step):
        vec = S[j * K/Step : (j+Step) * K/Step] * wind
        Spectrogram.append(abs(fft(vec, K)[:K/2]))

    return np.array(Spectrogram)

def plot_matrix(Sxx, title):
    cmap = grayify_cmap('cubehelix_r')
    #cmap = plt.cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Samples')
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


