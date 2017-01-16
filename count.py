import os
import glob

validation = "/disk/martinsson-spring17/birdClef2016Whole/valid"
train = "/disk/martinsson-spring17/birdClef2016Whole/train"

for (v, t) in zip(os.listdir(validation), os.listdir(train)):
    nb_valid = len(glob.glob(os.path.join(validation, v, "*.wav")))
    nb_train = len(glob.glob(os.path.join(train, t, "*.wav")))
    print(v, nb_valid/(nb_valid+nb_train))
