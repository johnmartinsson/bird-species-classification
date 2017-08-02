from functools import reduce
import glob
import os

import numpy as np

from bird import utils
from bird import signal_processing as sp
from bird import preprocessing as pp

# def gaussian_probability_density(x, mu, sigma):
    # p = 1/(2*np.pi*(sigma**2))**(1/2) * np.exp(-((x-mu)**2)/(2*sigma**2))
    # return p

def intersection(l1, l2):
    return list(set(l1).intersection(set(l2)))

def intersection_all(sets):
    return reduce(intersection, sets, sets[0])

def signal_energy(wave, samplerate):
    spectrogram = sp.wave_to_amplitude_spectrogram(wave, samplerate)
    return np.sum(spectrogram)

def signal_structure(wave, samplerate):
    spectrogram = sp.wave_to_amplitude_spectrogram(wave, samplerate)
    norm_spectrogram = pp.normalize(spectrogram)
    binary_image = pp.median_clipping(norm_spectrogram, 3)
    return np.sum(binary_image)

def compute_class_energy(class_path):
    files = glob.glob(os.path.join(class_path, "*.wav"))
    energies = [signal_energy(wave, samplerate) for (samplerate, wave) in
                map(utils.read_wave_file, files)]
    return sum(energies)

def compute_class_structure(class_path):
    files = glob.glob(os.path.join(class_path, "*.wav"))
    structures = [signal_structure(wave, samplerate) for (samplerate, wave) in
                map(utils.read_wave_file, files)]
    return sum(structures)
