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
    # if (s.getframerate() != 22050):
        # raise ValueError("Sampling rate of wave file should be 16000")

    strsig = s.readframes(s.getnframes())
    x = np.fromstring(strsig, np.short)
    fs = s.getframerate()
    s.close()

    x = x/32768.0

    return fs, x
