
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
SEED = 1337

from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K

from bird.models.resnet import ResNetBuilder
from bird.preprocessing.sound import SoundDataGenerator

batch_size = 128
nb_classes = 3
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 224, 1024
# number of channels
nb_channels = 3
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

model = ResNetBuilder.build_resnet_18((img_rows, img_cols, nb_channels), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = SoundDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './resnet_sound_data/train',  # this is the target directory
        target_size=(img_rows, img_cols),  # all images will be resized to 150x150
        batch_size=16,
        class_mode='categorical',
        save_to_dir='./resnet_sound_images',
        seed=SEED)  # since we use binary_crossentropy loss, we need binary labels

model.fit_generator(
        train_generator,
        samples_per_epoch=3,
        nb_epoch=1)

model.save_weights('resnet.h5')

# score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
