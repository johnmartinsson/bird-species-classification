from keras.layers import Input
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

def CubeRun(nb_classes, input_shape):
    """ Instantiate a CubeRun architecture

    # Arguments
        nb_classes: the number of classification classes

    # Returns
        A Keras model instance
    """

    # adapt input shape to the used backend
    #if K.image_dim_ordering() == 'th':
        #input_shape=(1, image_rows, image_cols)
    #else:
        #input_shape=(image_rows, image_cols, 1)

    img_input = Input(shape=input_shape)

    # adapt back normalization axis to the used backend
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3

    x = ZeroPadding2D((2, 2))(img_input)
    #x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    #x = Dropout(0.2)(x)

    # conv (64 5x5 kernels, stride size 2x1)
    x = Convolution2D(64, 5, 5, subsample=(2, 1))(x)
    x = Activation('relu')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis)(x)
    # conv (64 5x5 kernels, stride size 1x1)
    x = Convolution2D(64, 5, 5, subsample=(1, 1))(x)
    x = Activation('relu')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis)(x)
    # conv (128 5x5 kernels, stride size 1x1)
    x = Convolution2D(128, 5, 5, subsample=(1, 1))(x)
    x = Activation('relu')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis)(x)
    # conv (256 5x5 kernels, stride size 1x1)
    x = Convolution2D(256, 5, 5, activation="relu", subsample=(1, 1))(x)
    x = Activation('relu')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis)(x)
    # conv (256 3x3 kernels, stride size 1x1)
    x = Convolution2D(256, 3, 3, subsample=(1, 1))(x)
    x = Activation('relu')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # batch normalization
    #x = BatchNormalization(axis=bn_axis)(x)

    # flatten 3D feature maps to 1D feature vectors
    x = Flatten()(x)

    # dense layer
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    # dense layer dropout
    x = Dropout(0.4)(x)

    # soft max layer
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)
    # soft max layer dropout
    #x = Dropout(0.4)(x)

    model = Model(img_input, x)

    return model
