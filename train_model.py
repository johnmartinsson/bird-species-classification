from __future__ import print_function
import numpy as np
# np.random.seed(1337)  # for reproducibility
# SEED = 1337

import os
import pickle
from time import localtime, strftime
from optparse import OptionParser

import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD

from bird.models.cuberun import CubeRun
from bird.generators.sound import SoundDataGenerator

# Setup Callbacks for History
class HistoryCollector(keras.callbacks.Callback):
    def __init__(self, name):
        self.name = name
        self.data = []

    def on_epoch_end(self, batch, logs={}):
        self.data.append(logs.get(self.name))

def train_model(train_path, valid_path, batch_size, nb_classes, nb_epoch,
                nb_val_samples, samples_per_epoch, input_shape,
                weight_file_path, history_file_path, first_epoch, lock_file):

    img_rows, img_cols, nb_channels = input_shape

    model = CubeRun(nb_classes, input_shape)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    if first_epoch=='False':
        # load weights
        model.load_weights(weight_file_path)
        print("loading weigths from: " + weight_file_path)
    else:
        print("using initial weights")

    # Callback history collectors
    trainLossHistory = HistoryCollector('loss')
    validLossHistory = HistoryCollector('val_loss')
    trainAccHistory = HistoryCollector('acc')
    validAccHistory = HistoryCollector('val_acc')
    checkpoint = keras.callbacks.ModelCheckpoint(weight_file_path,
                                                monitor='val_acc', verbose=0,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='auto')

    # train data generator
    train_datagen = SoundDataGenerator(
        rescale=1./255,
        time_shift=True,
        pitch_shift=True,
        augment_with_same_class=True,
        augment_with_noise=True)

    # validation data generator
    valid_datagen = SoundDataGenerator(
        rescale=1./255)

    # Generator for training data
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
        #save_to_dir='./visuals/augmented_samples'
        )

    # Generator for validation data
    print("Loading validation data...")
    valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
        #save_to_dir='./visuals/validation_samples',
        )

    # Fit model to generated training data
    model.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch,
        validation_data=valid_generator,
        nb_val_samples=nb_val_samples,
        callbacks=[trainLossHistory, validLossHistory, trainAccHistory,
                   validAccHistory, checkpoint])

    # save the weights
    model.save_weights(weight_file_path)
    print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
    print ("The weights have been saved in: " + weight_file_path)

    # save history to file
    with open(history_file_path, 'wb') as output:
        pickle.dump(trainLossHistory.data, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validLossHistory.data, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainAccHistory.data, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validAccHistory.data, output, pickle.HIGHEST_PROTOCOL)
    print ("The history has been saved in: " + history_file_path)

    os.remove(lock_file)

    return weight_file_path

parser = OptionParser()
parser.add_option("--train_path", dest="train_path")
parser.add_option("--valid_path", dest="valid_path")
parser.add_option("--weight_path", dest="weight_path")
parser.add_option("--history_path", dest="history_path")
parser.add_option("--first_epoch", dest="first_epoch")
parser.add_option("--lock_file", dest="lock_file")

(options, args) = parser.parse_args()

batch_size = 16
nb_classes = 20
nb_epoch   = 10
nb_val_samples = 10
samples_per_epoch = 10
input_shape = (256, 512, 1)

train_path = options.train_path
valid_path = options.valid_path
history_file_path = options.history_path
weight_file_path = options.weight_path
first_epoch = options.first_epoch
lock_file = options.lock_file

train_model(train_path, valid_path, batch_size, nb_classes, nb_epoch,
            nb_val_samples, samples_per_epoch, input_shape, weight_file_path,
            history_file_path, first_epoch, lock_file)
