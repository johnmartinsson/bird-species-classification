#!/usr/bin/env python3

from __future__ import print_function
import numpy as np
# np.random.seed(1337)  # for reproducibility
# SEED = 1337

import os
import ast
import pickle
import configparser
from time import localtime, strftime
from optparse import OptionParser

import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD

from bird.models.cuberun import CubeRun
from bird.models.resnet import ResNetBuilder
from bird.generators.sound import SoundDataGenerator

# Setup Callbacks for History
class HistoryCollector(keras.callbacks.Callback):
    def __init__(self, name):
        self.name = name
        self.data = []

    def on_epoch_end(self, batch, logs={}):
        self.data.append(logs.get(self.name))

def train_model(config_file, weight_file_path, history_file_path, first_epoch, lock_file):

    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)

    # model
    batch_size = int(config_parser['MODEL']['BatchSize'])
    nb_classes = int(config_parser['MODEL']['NumberOfClasses'])
    nb_epoch = int(config_parser['MODEL']['NumberOfEpochs'])
    nb_iterations = int(config_parser['MODEL']['NumberOfIterations'])
    nb_val_samples = int(config_parser['MODEL']['NumberOfValidationSamplesPerEpoch'])
    samples_per_epoch = int(config_parser['MODEL']['NumberOfTrainingSamplesPerEpoch'])
    input_shape = ast.literal_eval(config_parser['MODEL']['InputShape'])
    model_name = config_parser['MODEL']['ModelName']
    audio_mode = config_parser['MODEL']['InputDataMode']

    # paths
    noise_path = config_parser['PATHS']['NoiseDataDir']
    train_path = config_parser['PATHS']['TrainingDataDir']
    valid_path = config_parser['PATHS']['ValidationDataDir']

    # training
    optimizer = config_parser['TRAINING']['Optimizer']
    learning_rate = float(config_parser['TRAINING']['LearningRate'])
    decay = float(config_parser['TRAINING']['Decay'])
    momentum = float(config_parser['TRAINING']['Momentum'])
    nesterov = config_parser['TRAINING']['Nesterov'] == 'True'
    loss_function = config_parser['TRAINING']['LossFunction']
    time_shift = config_parser['TRAINING']['TimeShiftAugmentation'] == 'True'
    pitch_shift = config_parser['TRAINING']['PitchShiftAugmentation'] == 'True'
    same_class_augmentation = config_parser['TRAINING']['SameClassAugmentation'] == 'True'
    noise_augmentation = config_parser['TRAINING']['NoiseAugmentation'] == 'True'

    img_rows, img_cols, nb_channels = input_shape

    model = None
    if model_name == 'cuberun':
	    model = CubeRun(nb_classes, input_shape)
    elif model_name == 'resnet_18':
            model = ResNetBuilder.build_resnet_18(input_shape, nb_classes)
    elif model_name == 'resnet_34':
            model = ResNetBuilder.build_resnet_34(input_shape, nb_classes)
    elif model_name == 'resnet_50':
            model = ResNetBuilder.build_resnet_50(input_shape, nb_classes)
    elif model_name == 'resnet_101':
            model = ResNetBuilder.build_resnet_101(input_shape, nb_classes)
    elif model_name == 'resnet_152':
            model = ResNetBuilder.build_resnet_152(input_shape, nb_classes)
    else:
        raise ValueError("Can not find model ", model_name, ".")

    if optimizer == 'sgd':
        sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum,
                  nesterov=nesterov)
        model.compile(loss=loss_function,
                      optimizer=sgd,
                      metrics=['accuracy'])
    else:
        model.compile(loss=loss_function,
                      optimizer=optimizer,
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
    best_weight_file_path = os.path.join(os.path.dirname(weight_file_path), "best_weights.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(best_weight_file_path,
                                                monitor='val_acc', verbose=0,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='auto')

    # train data generator
    train_datagen = SoundDataGenerator(
        rescale=1./255,
        time_shift=time_shift,
        pitch_shift=pitch_shift,
        augment_with_same_class=same_class_augmentation,
        augment_with_noise=noise_augmentation)

    # validation data generator
    valid_datagen = SoundDataGenerator(
        rescale=1./255)

    # Generator for training data
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_path,
        noise_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        audio_mode=audio_mode
        #save_to_dir='./visuals/augmented_samples'
        )

    # Generator for validation data
    print("Loading validation data...")
    valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        noise_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        audio_mode=audio_mode
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
parser.add_option("--config_file", dest="config_file")
parser.add_option("--history_path", dest="history_path")
parser.add_option("--weight_path", dest="weight_path")
parser.add_option("--first_epoch", dest="first_epoch")
parser.add_option("--lock_file", dest="lock_file")

(options, args) = parser.parse_args()
config_file = options.config_file
history_file_path = options.history_path
weight_file_path = options.weight_path
first_epoch = options.first_epoch
lock_file = options.lock_file

train_model(config_file, weight_file_path, history_file_path, first_epoch, lock_file)
