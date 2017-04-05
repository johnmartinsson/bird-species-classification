import numpy as np
SEED = 42
np.random.seed(SEED)

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

def get_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    # the model so far outputs 3D feature maps (height, width, features)

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def add_learning_scheme(model):
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

def get_data_train_generator():
    data_train_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        dim_ordering='tf')

    return data_train_generator

def get_data_validation_generator():
    data_validation_generator = ImageDataGenerator(
        rescale=1./255,
		dim_ordering='tf')

    return data_validation_generator

def add_data_train_flow(data_train_generator):
    data_train_generator.flow_from_directory(
        '../tutorial_data/train', # target directory
        target_size=(150, 150), # resize images to 150x150
        batch_size=32,
        class_mode='binary')

def add_data_validation_flow(data_validation_generator):
    data_validation_generator.flow_from_directory(
        '../tutorial_data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model = get_model()
add_learning_scheme(model)
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../tutorial_data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary',
        seed=SEED)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../tutorial_data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        seed=SEED)

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=1,
        validation_data=validation_generator,
        nb_val_samples=800)

model.save_weights('first_try.h5')
