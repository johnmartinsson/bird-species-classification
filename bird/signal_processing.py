import scipy
import numpy as np
import mock
import sys
import librosa

def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

def wave_to_complex_spectrogram(wave, fs):
    return librosa.stft(wave, n_fft=512, hop_length=128, win_length=512)

def wave_to_amplitude_spectrogram(wave, fs):
    X = wave_to_complex_spectrogram(wave, fs) #, framesamp, hopsamp)
    X = np.abs(X) ** 2
    return X[4:232]

def wave_to_log_amplitude_spectrogram(wave, fs):
    return np.log(wave_to_amplitude_spectrogram(wave, fs))

def wave_to_sample_spectrogram(wave, fs):
    # Han window of size 512, and hop size 128 (75% overlap)
    return wave_to_log_amplitude_spectrogram(wave, fs)

def wave_to_tempogram(wave, fs):
    tempogram = librosa.feature.tempogram(wave, fs)
    return tempogram
