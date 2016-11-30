import numpy as np
# Fix the random seed to make results reproducible
np.random.seed(42)

import os
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from time import localtime, strftime


from bird import loader as loader
from bird.models.cuberun import CubeRun

# Settings
nb_classes = 20
batch_size = 16
input_shape = (257, 512)
(image_height, image_width) = input_shape
#train_path = "./datasets/mlsp2013/train_preprocessed";
#labels_path = "./datasets/mlsp2013/train_preprocessed/file2labels.csv";
train_path = "./datasets/birdClef2016Subset_preprocessed/train";
train_labels_path = os.path.join(train_path, "file2labels.csv");
valid_path = "./datasets/birdClef2016Subset_preprocessed/valid";
valid_labels_path = os.path.join(valid_path, "file2labels.csv");
weight_file_path = "./weights/" + strftime("%Y_%m_%d_%H:%M:%S_", localtime()) + "cuberun.h5"
samplerate = 16000

# Settings Mini Batch Generator
nb_augmentation_samples = 10000
nb_mini_baches = 20
nb_epoch_per_mini_batch = 10
nb_segments_per_mini_batch = 300

model = CubeRun(nb_classes=nb_classes, input_shape=input_shape)

# Setup compile
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
# load the data
X_valid, Y_valid = loader.load_validation_data(valid_path, valid_labels_path, nb_classes,
                                        input_shape)


mini_batch_generator = loader.mini_batch_generator(nb_augmentation_samples,
                                              nb_mini_baches,
                                              nb_segments_per_mini_batch,
                                              train_path, train_labels_path,
                                                   nb_classes, samplerate)

# fit the model to training data
i_mini_batch = 1
for X_train, Y_train in mini_batch_generator:
    print("mini-batch: ", i_mini_batch, "/", nb_mini_baches)
    model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch_per_mini_batch, shuffle=True,
              validation_data=(X_valid, Y_valid))

    i_mini_batch += 1


# save the weights
model.save_weights(weight_file_path)
print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
print ("The weights have been saved in: " + weight_file_path)
