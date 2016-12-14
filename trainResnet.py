from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
SEED = 1337

from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD

from bird.models.resnet import ResNetBuilder
from bird.generators.sound import SoundDataGenerator

batch_size = 8
nb_classes = 20
samples_per_epoch = 2008
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 256, 512
# number of channels
nb_channels = 1

model = ResNetBuilder.build_resnet_18((img_rows, img_cols, nb_channels), nb_classes)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

train_datagen = SoundDataGenerator(
    rescale=1./255)
    #augment_with_same_class=True,
    #augment_with_noise=True)

valid_datagen = SoundDataGenerator(
    rescale=1./255)

# Generator for training data
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    './datasets/birdClef2016Subset/train',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    seed=SEED)

# Generator for validation data
print("Loading validation data...")
valid_generator = valid_datagen.flow_from_directory(
    './datasets/birdClef2016Subset/valid',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    seed=SEED)

# Fit model to generated training data
model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    validation_data=valid_generator,
    nb_val_samples=613)

# Save weights
model.save_weights('resnet.h5')

