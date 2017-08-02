""" This is a reimplementation of the work of Elias Sprengel
(http://ceur-ws.org/Vol-1609/16090547.pdf).
"""
from keras.layers import Input
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

def CubeRun(nb_classes, input_shape):
    """ Instantiate a CubeRun architecture

    # Arguments
        nb_classes  : the number of classification classes
        input_shape : the shape of the input layer (rows, columns)

    # Returns
        A Keras model instance
    """

    # adapt input shape to the used backend
    #(image_rows, image_cols) = input_shape
    #if K.image_dim_ordering() == 'th':
    #    input_shape=(1, image_rows, image_cols)
    #else:
    #    input_shape=(image_rows, image_cols, 1)

    img_input = Input(shape=input_shape)

    # adapt batch normalization axis to the used backend
    #if K.image_dim_ordering() == 'th':
    #    bn_axis = 1
    #else:
    bn_axis = 3

    # x = Dropout(0.2)(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(img_input)

    # conv (64 5x5 kernels, stride size 1x2)
    x = Convolution2D(64, 5, 5, subsample=(1, 2), activation='relu',
                      init="he_normal", border_mode="same", name='conv_1')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x)
    # conv (64 5x5 kernels, stride size 1x1)
    x = Convolution2D(64, 5, 5, subsample=(1, 1), activation='relu',
                      init="he_normal", border_mode="same", name='conv2')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis, name='bn_conv3')(x)
    # conv (128 5x5 kernels, stride size 1x1)
    x = Convolution2D(128, 5, 5, subsample=(1, 1), activation='relu',
                      init="he_normal", border_mode="same", name='conv3')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis, name='bn_conv4')(x)
    # conv (256 5x5 kernels, stride size 1x1)
    x = Convolution2D(256, 5, 5, subsample=(1, 1), activation='relu',
                      init="he_normal", border_mode="same", name='conv4')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # batch normalization
    x = BatchNormalization(axis=bn_axis, name='bn_conv5')(x)
    # conv (256 3x3 kernels, stride size 1x1)
    x = Convolution2D(256, 3, 3, subsample=(1, 1), activation='relu',
                      init="he_normal", border_mode="same", name='conv5')(x)
    # max pooling (2x2 kernels, stride size 2x2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = BatchNormalization(axis=bn_axis, name='bn_dense')(x)
    # flatten 3D feature maps to 1D feature vectors
    x = Flatten(name='flatten')(x)

    # dense layer
    x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu', name='dense')(x)

    # soft max layer
    x = Dropout(0.4)(x)
    x = Dense(nb_classes, activation='softmax', name='softmax')(x)

    model = Model(img_input, x)

    return model
