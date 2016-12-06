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
nb_epoch = 200
batch_size = 16

samplerate = 16000
input_shape = (253, 512)

train_path = "./datasets/birdClef2016Subset_preprocessed/train";
train_labels_path = os.path.join(train_path, "file2labels.csv");
valid_path = "./datasets/birdClef2016Subset_preprocessed/valid";
valid_labels_path = os.path.join(valid_path, "file2labels.csv");
weight_file_path = "./weights/" + strftime("%Y_%m_%d_%H:%M:%S_", localtime()) + "cuberun.h5"

model = CubeRun(nb_classes=nb_classes, input_shape=input_shape)

# Setup compile
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
# load the data
X_valid, Y_valid = loader.load_validation_data(valid_path, valid_labels_path, nb_classes)
X_train, Y_train = loader.load_validation_data(train_path, train_labels_path, nb_classes)


# fit the model to training data
model.fit(X_train, Y_train, batch_size=batch_size,
      nb_epoch=nb_epoch, shuffle=True,
      validation_data=(X_valid, Y_valid))

# save the weights
model.save_weights(weight_file_path)
print (strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
print ("The weights have been saved in: " + weight_file_path)
