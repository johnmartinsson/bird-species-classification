'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
import glob
import re
import scipy
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import mock
import sys
sys.modules.update((mod_name, mock.Mock()) for mod_name in ['matplotlib',
                                                            'matplotlib.pyplot',
                                                            'matplotlib.image'])
import librosa

from keras import backend as K

from bird import utils
from bird import data_augmentation as da
from bird import signal_processing as sp


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, dim_ordering='default', scale=True):
    from PIL import Image
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])

def load_wav_as_mfcc(fname, target_size=None, noise_files=None,
                       augment_with_noise=False, class_dir=None):
    (fs, signal) = utils.read_wave_file(fname)

    if class_dir:
        signal = da.same_class_augmentation(signal, class_dir)

    if augment_with_noise:
        signal = da.noise_augmentation(signal, noise_files)

    mfcc = librosa.feature.mfcc(signal, fs, n_mfcc=target_size[0])

    if target_size:
        mfcc= scipy.misc.imresize(mfcc, target_size)

    mfcc = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)

    return mfcc


def load_wav_as_mfcc_delta(fname, target_size=None, noise_files=None,
                           augment_with_noise=False, class_dir=None):
    (fs, signal) = utils.read_wave_file(fname)

    if class_dir:
        signal = da.same_class_augmentation(signal, class_dir)

    if augment_with_noise:
        signal = da.noise_augmentation(signal, noise_files)

    mfcc = librosa.feature.mfcc(signal, fs, n_mfcc=target_size[0])
    mfcc_delta_3 = librosa.feature.delta(mfcc, width=3, order=1)
    mfcc_delta_11 = librosa.feature.delta(mfcc, width=11, order=1)
    mfcc_delta_19 = librosa.feature.delta(mfcc, width=19, order=1)

    if target_size:
        mfcc = scipy.misc.imresize(mfcc, target_size)
        mfcc_delta_3 = scipy.misc.imresize(mfcc_delta_3, target_size)
        mfcc_delta_11 = scipy.misc.imresize(mfcc_delta_11, target_size)
        mfcc_delta_19 = scipy.misc.imresize(mfcc_delta_19, target_size)

    mfcc = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)
    mfcc_delta_3 = mfcc_delta_3.reshape(mfcc_delta_3.shape[0], mfcc_delta_3.shape[1], 1)
    mfcc_delta_11 = mfcc_delta_11.reshape(mfcc_delta_11.shape[0], mfcc_delta_11.shape[1], 1)
    mfcc_delta_19 = mfcc_delta_19.reshape(mfcc_delta_19.shape[0], mfcc_delta_19.shape[1], 1)
    mfcc_delta = np.concatenate([mfcc, mfcc_delta_3, mfcc_delta_11, mfcc_delta_19], axis=2)

    return mfcc_delta

def load_wav_as_spectrogram(fname, target_size=None, noise_files=None,
                       augment_with_noise=False, class_dir=None):
    (fs, signal) = utils.read_wave_file(fname)

    if class_dir:
        signal = da.same_class_augmentation(signal, class_dir)

    if augment_with_noise:
        signal = da.noise_augmentation(signal, noise_files)

    spectrogram = sp.wave_to_sample_spectrogram(signal, fs)

    if target_size:
        spectrogram = scipy.misc.imresize(spectrogram, target_size)

    spectrogram = spectrogram.reshape((spectrogram.shape[0],
                                       spectrogram.shape[1], 1))
    return spectrogram

def load_wav_as_tempogram(fname, target_size=None, noise_files=None,
                       augment_with_noise=False, class_dir=None):
    (fs, signal) = utils.read_wave_file(fname)

    if class_dir:
        signal = da.same_class_augmentation(signal, class_dir)

    if augment_with_noise:
        signal = da.noise_augmentation(signal, noise_files)

    tempogram = sp.wave_to_tempogram(signal, fs)

    if target_size:
        tempogram = scipy.misc.imresize(tempogram, target_size)

    tempogram = tempogram.reshape((tempogram.shape[0],
                                       tempogram.shape[1], 1))
    return tempogram



class SoundDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering='default',
                 augment_with_same_class=False,
                 augment_with_noise=False,
                 time_shift=False,
                 pitch_shift=False):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)
    def flow_from_directory(self, directory, noise_dir,
                            target_size=(256, 256), audio_mode='spectrogram',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        return DirectoryIterator(
            directory, noise_dir, self,
            target_size=target_size, audio_mode=audio_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class DirectoryIterator(Iterator):

    def __init__(self, directory, noise_dir, image_data_generator,
                 target_size=(256, 256), audio_mode='spectrogram',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        # Cache noise sets
        noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
        random.shuffle(noise_files)
        self.noise_files = noise_files
        print("Loaded ", len(noise_files), " noise segments")
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if audio_mode not in {'spectrogram', 'tempogram', 'mfcc', 'mfcc_delta'}:
            raise ValueError('Invalid audio mode:', audio_mode,
                             '; expected "spectrogram", "tempogram", "mfcc" or "mfcc_delta".')
        self.audio_mode = audio_mode
        self.dim_ordering = dim_ordering
        if self.audio_mode == 'mfcc_delta':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.audio_mode in {'mfcc', 'spectrogram', 'tempogram'}:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'wav'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in sorted(os.listdir(subpath)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
        print('Found %d sounds belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in sorted(os.listdir(subpath)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.classes[i] = self.class_indices[subdir]
                    self.filenames.append(os.path.join(subdir, fname))
                    i += 1
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]

            class_dir = None
            if self.image_data_generator.augment_with_same_class:
                class_dir = os.path.join(self.directory, os.path.split(fname)[0])

            if self.audio_mode == 'spectrogram':
                x = load_wav_as_spectrogram(os.path.join(self.directory, fname),
                                       target_size=self.target_size,
                                       noise_files=self.noise_files,
                                       augment_with_noise=self.image_data_generator.augment_with_noise, class_dir=class_dir)

            elif self.audio_mode == 'tempogram':
                x = load_wav_as_tempogram(os.path.join(self.directory, fname),
                                       target_size=self.target_size,
                                       noise_files=self.noise_files,
                                       augment_with_noise=self.image_data_generator.augment_with_noise, class_dir=class_dir)

            elif self.audio_mode == 'mfcc':
                x = load_wav_as_mfcc(os.path.join(self.directory, fname),
                                           target_size=self.target_size,
                                           noise_files=self.noise_files,
                                           augment_with_noise=self.image_data_generator.augment_with_noise,
                                           class_dir=class_dir)
            elif self.audio_mode == 'mfcc_delta':
                x = load_wav_as_mfcc_delta(os.path.join(self.directory, fname),
                                           target_size=self.target_size,
                                           noise_files=self.noise_files,
                                           augment_with_noise=self.image_data_generator.augment_with_noise,
                                           class_dir=class_dir)

            if self.image_data_generator.time_shift:
                x = da.time_shift_spectrogram(x)
            if self.image_data_generator.pitch_shift:
                x = da.pitch_shift_spectrogram(x)

            # x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
