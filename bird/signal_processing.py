import scipy
import numpy as np

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

def wave_to_complex_spectrogram(wave, fs, framesamp, hopsamp):
    framesz = framesamp/fs
    hop = hopsamp/fs
    return stft(wave, fs, framesz, hop).T

def wave_to_amplitude_spectrogram(wave, fs, framesamp, hopsamp):
    X = wave_to_complex_spectrogram(wave, fs, framesamp, hopsamp)
    X = np.abs(X) ** 2
    return X[4:232]

def wave_to_log_amplitude_spectrogram(wave, fs, framesamp, hopsamp):
    return np.log(wave_to_amplitude_spectrogram(wave, fs, framesamp, hopsamp))

def wave_to_sample_spectrogram(wave, fs):
    # Han window of size 512, and hop size 128 (75% overlap)
    return wave_to_log_amplitude_spectrogram(wave, fs, 512, 128)

