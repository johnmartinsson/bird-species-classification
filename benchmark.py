from bird import preprocessing as pp
from bird import utils

filename = "datasets/mlsp2013/train/PC10_20090513_054500_0020.wav.gz"
(fs, x) = utils.read_gzip_wave_file(filename)
(t, f, Sxx) = utils.wave_to_spectrogram(x, fs)
n_mask = pp.compute_signal_mask(Sxx)
n_mask_scaled = pp.reshape_binary_mask(n_mask, x.shape[0])
Nxx = pp.normalize(Sxx)

def test():
    "Stupid test function"
    L = [i for i in range(100)]

def compute_spectrogram():
	utils.wave_to_spectrogram(x, fs)

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

def read_gzip_wave_file():
	utils.read_gzip_wave_file(filename)

def median_clipping():
	pp.median_clipping(Nxx, 3.0)

if __name__=='__main__':
	import timeit
	number = 10
	print("compute_spectrogram():", timeit.timeit("compute_spectrogram()", setup="from __main__ import compute_spectrogram", number=number))
	print("read_gzip_wave_file():", timeit.timeit("read_gzip_wave_file()", setup="from __main__ import read_gzip_wave_file", number=number))
	print("median_clipping():", timeit.timeit("median_clipping()", setup="from __main__ import median_clipping", number=number))
	print("compute_noise_mask():", timeit.timeit("compute_noise_mask()", setup="from __main__ import compute_noise_mask", number=number))
	print("compute_signal_mask():", timeit.timeit("compute_signal_mask()", setup="from __main__ import compute_signal_mask", number=number))
	print("reshape_binary_mask():", timeit.timeit("reshape_binary_mask()", setup="from __main__ import reshape_binary_mask", number=number))
	print("extract_masked_part_from_wave():", timeit.timeit("extract_masked_part_from_wave()", setup="from __main__ import extract_masked_part_from_wave", number=number))
	print("preprocess_wave():", timeit.timeit("preprocess_wave()", setup="from __main__ import preprocess_wave", number=number))
