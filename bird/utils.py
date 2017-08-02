import numpy as np
import os
import csv
import glob
import sys
import subprocess
import wave
import gzip
import shutil

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

def read_wave_file_not_normalized(filename):
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

    return fs, x



def copy_subset(root_dir, classes, subset_dir):
    # create directories
    if not os.path.exists(subset_dir):
        print("os.makedirs("+subset_dir+")")
        os.makedirs(subset_dir)
    subset_dir_valid = os.path.join(subset_dir, "valid")
    subset_dir_train = os.path.join(subset_dir, "train")
    if not os.path.exists(subset_dir_valid):
        print("os.makedirs("+subset_dir_valid+")")
        os.makedirs(subset_dir_valid)
    if not os.path.exists(subset_dir_train):
        print("os.makedirs("+subset_dir_train+")")
        os.makedirs(subset_dir_train)

    for c in classes:
        valid_source_dir = os.path.join(root_dir, "valid", c)
        train_source_dir = os.path.join(root_dir, "train", c)
        valid_dest_dir = os.path.join(subset_dir_valid, c)
        train_dest_dir = os.path.join(subset_dir_train, c)

        print("shutil.copytree(" + valid_source_dir + "," + valid_dest_dir + ")")
        shutil.copytree(valid_source_dir, valid_dest_dir)
        print("shutil.copytree(" + train_source_dir + "," + train_dest_dir + ")")
        shutil.copytree(train_source_dir, train_dest_dir)
