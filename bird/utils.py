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

def noise_mask(spectrogram):
    print("noise_mask is undefined")

def structure_mask(spectrogram):
    print("structure_mask is undefined")

def extract_signal(mask, spectrogram):
    print("extract_signal is undefined")

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

def wave_to_spectrogram(wave=np.array([]), fs=None, window=signal.hanning(512),
                       nperseg=512, noverlap=256):
    """Given a wave form returns the spectrogram of the wave form.

    Keyword arguments:
    wave -- the wave form (default np.array([]))
    fs   -- the rate at which the wave form has been sampled
    """
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

def show_spectrogram(Sxx):
    plt.pcolor(Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()
