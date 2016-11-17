import numpy as np
import os
import sys
import subprocess
import wave
import wave
from scipy import signal
from scipy import fft
from matplotlib import pyplot as plt

MLSP_DATA_PATH="/home/darksoox/gits/bird-species-classification/mlsp_contest_dataset/"

def play_wave_file(filename):
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")
    else:
        if (sys.platform == "linux" or sys.playform == "linux2"):
            subprocess.call(["aplay", filename])
        else:
            print("Platform not supported")

def read_wave_file(filename):

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

    Keyword arguments:
    wave -- the wave form (default np.array([]))
    fs   -- the rate at which the wave form has been sampled
    """
    window = signal.get_window('hanning', nperseg)
    return signal.spectrogram(wave, fs, window, nperseg, noverlap,
                              mode='magnitude')

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

def plot_matrix(Sxx):
    cmap = grayify_cmap('cubehelix_r')
    #cmap = plt.cm.get_cmap('gist_rainbow')
    plt.figure()
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
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
