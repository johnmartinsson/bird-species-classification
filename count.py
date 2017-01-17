import os
import glob
from bird import loader as l

validation = "./datasets/birdClef2016Whole/valid"
train = "./datasets/birdClef2016Whole/train"

for (v, t) in zip(os.listdir(validation), os.listdir(train)):
    nb_valid = len(l.group_segments(glob.glob(os.path.join(validation, v, "*.wav"))))
    nb_train = len(l.group_segments(glob.glob(os.path.join(train, t, "*.wav"))))
    print(v, nb_valid/(nb_valid+nb_train))
