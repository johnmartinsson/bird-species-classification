import os
import glob

validation = "./datasets/birdClef2016Subset/valid"
train = "./datasets/birdClef2016Subset/train"
for (v, t) in zip(os.listdir(validation), os.listdir(train)):
    nb_valid = len(glob.glob(os.path.join(validation, v, "*.wav")))
    nb_train = len(glob.glob(os.path.join(train, t, "*.wav")))
    print(v, nb_valid/(nb_valid+nb_train))
