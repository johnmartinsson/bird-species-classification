import glob
import os

import numpy as np

from bird import utils
from bird import signal_processing as sp

def intersection(l1, l2):
    return list(set(l1).intersection(set(l2)))

def signal_energy(wave, samplerate):
    spectrogram = sp.wave_to_amplitude_spectrogram(wave, samplerate)
    return np.sum(spectrogram)

def compute_class_energy(class_path):
    files = glob.glob(os.path.join(class_path, "*.wav"))
    energies = [signal_energy(wave, samplerate) for (samplerate, wave) in
                map(utils.read_wave_file, files)]
    return sum(energies)
