import numpy as np
import os
import csv
import glob
import sys
import subprocess
import wave
import tqdm
import gzip

from scipy import signal
from scipy import fft
from scipy.io import wavfile
#from matplotlib import pyplot as plt
from functools import reduce

from bird import preprocessing as pp
from bird import loader as loader

def get_basename_without_ext(filepath):
    basename = os.path.basename(filepath).split(os.extsep)[0]
    return basename

def plot_spectrogram_from_wave(filename):
    fs, x = read_gzip_wave_file(filename)
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

def read_gzip_wave_file(filename):
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")

    with gzip.open(filename, 'rb') as wav_file:
        with wave.open(wav_file, 'rb') as s:
            if (s.getnchannels() != 1):
                raise ValueError("Wave file should be mono")
            if (s.getframerate() != 16000):
                raise ValueError("Sampling rate of wave file should be 16000")

            strsig = s.readframes(s.getnframes())
            x = np.fromstring(strsig, np.short)
            fs = s.getframerate()
            s.close()

            return fs, x


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
def wave_to_spectrogram_aux(wave, fs):
    (f, t, Sxx) = wave_to_spectrogram(wave, fs)
    return Sxx

def wave_to_log_spectrogram_aux(wave, fs):
    """ Compute a log magnitude spectrogram from the given signal
    """
    Sxx = wave_to_spectrogram_aux(wave, fs)
    return np.log(Sxx)

def compute_and_save_mask_as_image_from_file(filename):
    fs, x = read_gzip_wave_file(filename)
    t, f, Sxx = wave_to_spectrogram(x, fs)
    basename = get_basename_without_ext(filename)
    mask = pp.compute_binary_mask(Sxx, 3.0, True, basename+".png")

#def subplot_image(Sxx, n_subplot, title):
    #cmap = grayify_cmap('cubehelix_r')
    #plt.subplot(n_subplot)
    #plt.title(title)
    #plt.pcolormesh(Sxx, cmap=cmap)

#def save_matrix_to_file(Sxx, title, filename):
    #cmap = grayify_cmap('cubehelix_r')
    ##cmap = plt.cm.get_cmap('gist_rainbow')
    #fig = plt.figure()
    #fig.suptitle(title, fontsize=12)
    #plt.pcolormesh(Sxx, cmap=cmap)
    #plt.ylabel('Frequency Bins')
    #plt.xlabel('Samples')
    #fig.savefig(filename)

#def plot_matrix(Sxx, title):
    #cmap = grayify_cmap('cubehelix_r')
    ##cmap = plt.cm.get_cmap('gist_rainbow')
    #fig = plt.figure()
    #fig.suptitle(title, fontsize=12)
    #plt.pcolormesh(Sxx, cmap=cmap)
    #plt.ylabel('Frequency Bins')
    #plt.xlabel('Samples')
    #plt.show()

#def plot_vector(x):
    #mesh = np.zeros((257, x.shape[0]))
    #for i in range(x.shape[0]):
        #mesh[256][i] = x[i] * 2500
        #mesh[255][i] = x[i] * 2500
        #mesh[254][i] = x[i] * 2500
    #plot_matrix(mesh)

#def grayify_cmap(cmap):
    #"""Return a grayscale version of the colormap"""
    #cmap = plt.cm.get_cmap(cmap)
    #colors = cmap(np.arange(cmap.N))

    ## convert RGBA to perceived greyscale luminance
    ## cf. http://alienryderflex.com/hsp.html
    #RGB_weight = [0.299, 0.587, 0.114]
    #luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    #colors[:, :3] = luminance[:, np.newaxis]

#    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)
