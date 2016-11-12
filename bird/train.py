from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from models.cuberun import CubeRun
from time import localtime, strftime
import loader

# Settings
nb_epoch = 5
#nb_classes = 19
nb_classes = 2
batch_size = 8
#image_shape = (257, 624)
input_shape = (256, 256, 3)
(cols, rows, dims) = input_shape
train_path = "../datasets/mlsp2013/train";
labels_path = "../datasets/mlsp2013/train/file2labels.csv";

model = CubeRun(nb_classes=nb_classes, input_shape=input_shape)

datagen = ImageDataGenerator(
    featurewise_center=False, # Boolean. Set input mean to 0 over the dataset.
    samplewise_center=False, # Boolean. Set each sample mean to 0.
    featurewise_std_normalization=False, # Boolean. Divide inputs by std of the dataset.
    samplewise_std_normalization=False, # Boolean. Divide each input by its std.
    zca_whitening=False, # Boolean. Apply ZCA whitening.
    rotation_range=0., # Int. Degree range for random rotations.
    width_shift_range=0., # Float (fraction of total width). Range for random horizontal shifts.
    height_shift_range=0., # Float (fraction of total height). Range for random vertical shifts.
    shear_range=0., # Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
    zoom_range=0., # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
    channel_shift_range=0., # Float. Range for random channel shifts.
    fill_mode='nearest', # One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode.
    cval=0., # Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
    horizontal_flip=False, # Boolean. Randomly flip inputs horizontally.
    vertical_flip=False, # Boolean. Randomly flip inputs vertically.
    rescale=None, # rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
    dim_ordering=K.image_dim_ordering())


############################################################################
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
training_class_dict = train_generator = train_datagen.flow_from_directory(
        '../tutorial_data/train',  # this is the target directory
        target_size=(cols, rows),  # all images will be resized to cols x rows
        batch_size=32,
        #classes=['cats', 'dogs'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_class_dict = validation_generator = test_datagen.flow_from_directory(
        '../tutorial_data/validation',
        target_size=(cols, rows),
        batch_size=32,
        #classes=['cats', 'dogs'],
        class_mode='categorical')

print("Training class dict: ", training_class_dict.class_indices)
print("Validation class dict: ", validation_class_dict.class_indices)
############################################################################

#X_valid, Y_valid = loader.load_data(train_path, labels_path, 100)


# Setup compile
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


############################################################################
model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=50,
    validation_data=validation_generator,
    nb_val_samples=800)

############################################################################

#for e in range(nb_epoch):
    #print("epoch %d" % e)
print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
#X_train, Y_train = loader.load_all_data(train_path, labels_path);
print(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
    #datagen.fit(X_train)
    #model.fit(X_train, Y_train, batch_size=16, nb_epoch=nb_epoch)
#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
      #verbose=1, validation_data=(X_valid, Y_valid))
    #for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=16):
        #loss = model.train(X_batch, Y_batch)
        #print("Loss: ", loss)

model.save_weights("../weights/cat_dog_cuberun.h5")
