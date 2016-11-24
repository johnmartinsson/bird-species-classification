import numpy as np

# Fix the random seed to make results reproducible
np.random.seed(42)

def time_shift(spectrogram):
    return

def pitch_shift(spectrogram):
    return

def additively_combine_segments(spectrogram_segments):
    return

def find_same_class_files(file2labels, labels):
    same_class_files = []
    for key, value in file2labels.items():
        if(labels == value):
            same_class_files.append(key)
    return same_class_files

def same_class_augmentation(s1):
    return
