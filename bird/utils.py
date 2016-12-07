import numpy as np
import os
import csv
import glob
import sys
import subprocess
import wave
import gzip

from scipy import signal
from scipy import fft
from scipy.io import wavfile
from functools import reduce

from bird import preprocessing as pp
from bird import loader as loader

def get_basename_without_ext(filepath):
    basename = os.path.basename(filepath).split(os.extsep)[0]
    return basename

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
            #if (s.getframerate() != 22050):
                #raise ValueError("Sampling rate of wave file should be 16000")

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
    if (s.getframerate() != 22050):
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
    return np.log10(Sxx + 0.001)
