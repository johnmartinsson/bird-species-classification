from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
SEED = 1337

import os
import pickle
from time import localtime, strftime

import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD

from bird.models.resnet import ResNetBuilder
from bird.generators.sound import SoundDataGenerator

train_path = "/disk/martinsson-spring17/birdClef2016Subset/train";
valid_path = "/disk/martinsson-spring17/birdClef2016Subset/valid";

#train_path = "/disk/martinsson-spring17/birdClef2016Whole/train";
#valid_path = "/disk/martinsson-spring17/birdClef2016Whole/valid";

basename = strftime("%Y_%m_%d_%H:%M:%S_", localtime()) + "resnet"
weight_file_path = os.path.join("./weights", basename + ".h5")
history_file_path = os.path.join("./history", basename + ".pkl")
print ("The weights are saved in: " + weight_file_path)

batch_size = 16
nb_classes = 20
samples_per_epoch = 2113
nb_val_samples=613
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 256, 512
# number of channels
nb_channels = 1

# Setup Callbacks for History
class HistoryCollector(keras.callbacks.Callback):
    def __init__(self, name):
        self.name = name
        self.data = []

    def on_epoch_end(self, batch, logs={}):
        self.data.append(logs.get(self.name))

trainLossHistory = HistoryCollector('loss')
validLossHistory = HistoryCollector('val_loss')
trainAccHistory = HistoryCollector('acc')
validAccHistory = HistoryCollector('val_acc')

checkpoint = keras.callbacks.ModelCheckpoint(weight_file_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')

model = ResNetBuilder.build_resnet_34((img_rows, img_cols, nb_channels), nb_classes)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

train_datagen = SoundDataGenerator(
    rescale=1./255,
    time_shift=True,
    pitch_shift=True,
    augment_with_same_class=True,
    augment_with_noise=True)

valid_datagen = SoundDataGenerator(
    rescale=1./255)

# Generator for training data
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    #save_to_dir='./visuals/augmented_samples',
    seed=SEED)

# Generator for validation data
print("Loading validation data...")
valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    #save_to_dir='./visuals/validation_samples',
    seed=SEED)

# Fit model to generated training data
model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    validation_data=valid_generator,
    nb_val_samples=nb_val_samples,
    callbacks=[trainLossHistory, validLossHistory, trainAccHistory, validAccHistory, checkpoint])

# save the weights
# model.save_weights(weight_file_path)
print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

# save history to file
with open(history_file_path, 'wb') as output:
    pickle.dump(trainLossHistory.data, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(validLossHistory.data, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(trainAccHistory.data, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(validAccHistory.data, output, pickle.HIGHEST_PROTOCOL)
print ("The history has been saved in: " + history_file_path)
