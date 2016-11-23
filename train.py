from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from time import localtime, strftime

import numpy as np

from bird import loader as loader
from bird.models.cuberun import CubeRun

# Settings
nb_epoch = 20
nb_classes = 19
batch_size = 8
input_shape = (257, 509)
(image_height, image_width) = input_shape
train_path = "./datasets/mlsp2013/train_preprocessed";
labels_path = "./datasets/mlsp2013/train_preprocessed/file2labels.csv";
weight_file_path = "./weights/" + strftime("%Y_%m_%d_%H:%M:%S_", localtime()) + "cuberun.h5"

model = CubeRun(nb_classes=nb_classes, input_shape=input_shape)

# Setup compile
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
# load the data
X, Y, filenames = loader.load_all_data(train_path, labels_path,
                            nb_classes=nb_classes,
                            image_shape=(image_width, image_height));

nb_x = X.shape[0]
nb_y = Y.shape[0]
nb_x_train = int(np.floor(0.7 * nb_x))
nb_y_train = int(np.floor(0.7 * nb_y))
# split data into training and validation set
X_train = X[:nb_x_train]
Y_train = Y[:nb_y_train]
X_valid = X[nb_x_train:]
Y_valid = Y[nb_x_train:]

print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
print ("X_train shape: ", X_train.shape)
print ("Y_train shape: ", Y_train.shape)

# fit the model to training data
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
      verbose=1, validation_data=(X_valid, Y_valid))
print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

model.save_weights(weight_file_path)
print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
print ("The weights have been saved in: " + weight_file_path)

#datagen = ImageDataGenerator(
    #featurewise_center=False, # Boolean. Set input mean to 0 over the dataset.
    #samplewise_center=False, # Boolean. Set each sample mean to 0.
    #featurewise_std_normalization=False, # Boolean. Divide inputs by std of the dataset.
    #samplewise_std_normalization=False, # Boolean. Divide each input by its std.
    #zca_whitening=False, # Boolean. Apply ZCA whitening.
    #rotation_range=0., # Int. Degree range for random rotations.
    #width_shift_range=0., # Float (fraction of total width). Range for random horizontal shifts.
    #height_shift_range=0., # Float (fraction of total height). Range for random vertical shifts.
    #shear_range=0., # Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
    #zoom_range=0., # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
    #channel_shift_range=0., # Float. Range for random channel shifts.
    #fill_mode='nearest', # One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode.
    #cval=0., # Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
    #horizontal_flip=False, # Boolean. Randomly flip inputs horizontally.
    #vertical_flip=False, # Boolean. Randomly flip inputs vertically.
    #rescale=None, # rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
    #dim_ordering=K.image_dim_ordering())


############################################################################
# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
#test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
#training_class_dict = train_generator = train_datagen.flow_from_directory(
        #'../tutorial_data/train',  # this is the target directory
        #target_size=(cols, rows),  # all images will be resized to cols x rows
        #batch_size=32,
        ##classes=['cats', 'dogs'],
        #class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
#validation_class_dict = validation_generator = test_datagen.flow_from_directory(
        #'../tutorial_data/validation',
        #target_size=(cols, rows),
        #batch_size=32,
        ##classes=['cats', 'dogs'],
        #class_mode='categorical')

#print ("Training class dict: ", training_class_dict.class_indices)
#print ("Validation class dict: ", validation_class_dict.class_indices)
############################################################################

#X_valid, Y_valid = loader.load_data(train_path, labels_path, size=100,
                                    #nb_classes=nb_classes, image_shape=(cols, rows))



############################################################################
#model.fit_generator(
    #train_generator,
    #samples_per_epoch=2000,
    #nb_epoch=50,
    #validation_data=validation_generator,
    #nb_val_samples=800)
############################################################################

#for e in range(nb_epoch):
    #print "epoch %d" % e
