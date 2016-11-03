from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()

# conv (64 5x5 kernels, stride size 2x1)
# TODO : 1 channel?
model.add(Convolution2D(64, 5, 5, input_shape=(1, 128, 256), activation="relu", subsample=(2, 1)))
# max pooling (2x2 kernels, stride size 2x2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# conv (64 5x5 kernels, stride size 1x1)
model.add(Convolution2D(64, 5, 5, activation="relu", subsample=(1, 1)))
# max pooling (2x2 kernels, stride size 2x2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# conv (128 5x5 kernels, stride size 1x1)
model.add(Convolution2D(128, 5, 5, activation="relu", subsample=(1, 1)))
# max pooling (2x2 kernels, stride size 2x2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# conv (256 5x5 kernels, stride size 1x1)
model.add(Convolution2D(256, 5, 5, activation="relu", subsample=(1, 1)))
# max pooling (2x2 kernels, stride size 2x2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# conv (256 3x3 kernels, stride size 1x1)
model.add(Convolution2D(256, 3, 3, activation="relu", subsample=(1, 1)))
# max pooling (2x2 kernels, stride size 2x2)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# dense (1024 units)
model.add(Dense(1024, input_shape=()))
# soft max (19 units)
model.add(Dense(19, activation='softmax'))

# TODO: compile model
