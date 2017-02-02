import glob
import random

from bird import preprocessing as pp
from bird import signal_processing as sp
from bird import data_augmentation as da
import bird.generators.sound as gs
from bird import utils

filename = "/disk/martinsson-spring17/datasets/birdClef2016Subset/train/affinis/LIFECLEF2015_BIRDAMAZON_XC_WAV_RN14132_seg_0.wav"
(fs, x) = utils.read_wave_file(filename)
Sxx = sp.wave_to_sample_spectrogram(x, fs)
n_mask = pp.compute_signal_mask(Sxx)
n_mask_scaled = pp.reshape_binary_mask(n_mask, x.shape[0])
Nxx = pp.normalize(Sxx)

target_size = (256, 512)
noise_files = glob.glob("/disk/martinsson-spring17/birdClef2016Whole/noise/*.wav")
noise_files_small = glob.glob("/home/martinsson-spring17/data/noise/*.wav")
class_dir = "/disk/martinsson-spring17/datasets/birdClef2016Subset/train/affinis"

def compute_tempogram():
    sp.wave_to_tempogram(x, fs)
def compute_spectrogram():
    sp.wave_to_sample_spectrogram(x, fs)

def extract_signal_part():
    pp.extract_signal_part(Sxx)

def extract_noise_part():
    pp.extract_noise_part(Sxx)

def extract_masked_part_from_wave():
    pp.extract_masked_part_from_wave(n_mask_scaled, x)

def compute_noise_mask():
    pp.compute_noise_mask(Sxx)

def compute_signal_mask():
    pp.compute_signal_mask(Sxx)

def reshape_binary_mask():
    pp.reshape_binary_mask(n_mask, 160000)

def preprocess_wave():
    pp.preprocess_wave(x, fs)

def read_wave_file():
    utils.read_wave_file(filename)

def median_clipping():
    pp.median_clipping(Nxx, 3.0)

def load_wav_as_spectrogram():
    gs.load_wav_as_spectrogram(filename, target_size, noise_files, class_dir)
def load_wav_as_spectrogram_small():
    gs.load_wav_as_spectrogram(filename, target_size, noise_files_small, class_dir)

def load_wav_as_mfcc_delta():
    gs.load_wav_as_mfcc_delta(filename, target_size, noise_files, class_dir)
def load_wav_as_mfcc_delta_small():
    gs.load_wav_as_mfcc_delta(filename, target_size, noise_files_small, class_dir)

def load_wav_as_mfcc():
    gs.load_wav_as_mfcc(filename, target_size, noise_files, class_dir)
def load_wav_as_mfcc_small():
    gs.load_wav_as_mfcc(filename, target_size, noise_files_small, class_dir)

def same_class_augmentation():
    da.same_class_augmentation(x, class_dir)

def read_random_noise_file():
    f = random.choice(noise_files)
    (fs, x) = utils.read_wave_file(f)
    x = x * 2

def read_random_noise_file_small():
    f = random.choice(noise_files_small)
    (fs, x) = utils.read_wave_file(f)
    x = x * 2

def noise_augmentation():
    da.noise_augmentation(x, noise_files_small)

def choose_noise_segments():
    nb_noise_segments = 3
    aug_noise_files = []
    for i in range(nb_noise_segments):
        aug_noise_files.append(random.choice(noise_files_small))
    f1 = aug_noise_files[0] + "h"
    f2 = aug_noise_files[1] + "j"
    f3 = aug_noise_files[2] + "k"

def choose_noise_segments_small():
    nb_noise_segments = 3
    aug_noise_files = []
    for i in range(nb_noise_segments):
        aug_noise_files.append(random.choice(noise_files_small))
    f1 = aug_noise_files[0] + "h"
    f2 = aug_noise_files[1] + "j"
    f3 = aug_noise_files[2] + "k"

def compute_noise_augmented():
    nb_noise_segments = 3
    aug_noise_files = []
    wave = x
    for i in range(nb_noise_segments):
        aug_noise_files.append(random.choice(noise_files_small))
    dampening_factor = 0.4
    for aug_noise_path in aug_noise_files:
        (fs, aug_noise) = utils.read_wave_file(aug_noise_path)
        wave = wave + aug_noise*dampening_factor
    return wave

if __name__=='__main__':
    import timeit
    number = 100
    print("compute_spectrogram():", timeit.timeit("compute_spectrogram()", setup="from __main__ import compute_spectrogram", number=number))
    print("compute_tempogram():", timeit.timeit("compute_tempogram()", setup="from __main__ import compute_tempogram", number=number))
    print("read_wave_file():", timeit.timeit("read_wave_file()", setup="from __main__ import read_wave_file", number=number))
    print("median_clipping():", timeit.timeit("median_clipping()", setup="from __main__ import median_clipping", number=number))
    print("compute_noise_mask():", timeit.timeit("compute_noise_mask()", setup="from __main__ import compute_noise_mask", number=number))
    print("compute_signal_mask():", timeit.timeit("compute_signal_mask()", setup="from __main__ import compute_signal_mask", number=number))
    print("reshape_binary_mask():", timeit.timeit("reshape_binary_mask()", setup="from __main__ import reshape_binary_mask", number=number))
    print("extract_masked_part_from_wave():", timeit.timeit("extract_masked_part_from_wave()", setup="from __main__ import extract_masked_part_from_wave", number=number))
    print("preprocess_wave():", timeit.timeit("preprocess_wave()", setup="from __main__ import preprocess_wave", number=number))
    print("load_wav_as_spectrogram():", timeit.timeit("load_wav_as_spectrogram()", setup="from __main__ import load_wav_as_spectrogram", number=number))
    print("load_wav_as_spectrogram_small():", timeit.timeit("load_wav_as_spectrogram_small()", setup="from __main__ import load_wav_as_spectrogram_small", number=number))
    print("load_wav_as_mfcc_delta():", timeit.timeit("load_wav_as_mfcc_delta()", setup="from __main__ import load_wav_as_mfcc_delta", number=number))
    print("load_wav_as_mfcc_delta_small():", timeit.timeit("load_wav_as_mfcc_delta_small()", setup="from __main__ import load_wav_as_mfcc_delta_small", number=number))
    print("load_wav_as_mfcc():", timeit.timeit("load_wav_as_mfcc()", setup="from __main__ import load_wav_as_mfcc", number=number))
    print("load_wav_as_mfcc_small():", timeit.timeit("load_wav_as_mfcc_small()", setup="from __main__ import load_wav_as_mfcc_small", number=number))
    print("read_random_noise_file():", timeit.timeit("read_random_noise_file()", setup="from __main__ import read_random_noise_file", number=number))
    print("read_random_noise_file_small():", timeit.timeit("read_random_noise_file_small()", setup="from __main__ import read_random_noise_file_small", number=number))
    print("same_class_augmentation():", timeit.timeit("same_class_augmentation()", setup="from __main__ import same_class_augmentation", number=number))
    print("noise_augmentation():", timeit.timeit("noise_augmentation()", setup="from __main__ import noise_augmentation", number=number))
    print("choose_noise_segments():", timeit.timeit("choose_noise_segments()", setup="from __main__ import choose_noise_segments", number=number))
    print("choose_noise_segments_small():", timeit.timeit("choose_noise_segments_small()", setup="from __main__ import choose_noise_segments_small", number=number))
    print("compute_noise_augmented():", timeit.timeit("compute_noise_augmented()", setup="from __main__ import compute_noise_augmented", number=number))
