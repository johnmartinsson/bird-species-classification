import glob
import os
import tqdm
import numpy as np
import shutil
from bird import loader as l

source_dir = "/disk/martinsson-spring17/birdClef2016Subset"
classes = os.listdir(os.path.join(source_dir, "train"))

percentage_validation_sampels = 0.10

progress = tqdm.tqdm(range(len(classes)))
class_segmentss = [(c, glob.glob(os.path.join(source_dir, "train", c, "*.wav"))) for c
                 in classes]
unique_sampels = [(c, l.group_segments(class_segments)) for (c, class_segments) in
                  class_segmentss]

print("Found ", sum([len(segs) for (c, segs) in unique_sampels]), " unique sampels")

for ((c, segments), p) in zip(unique_sampels, progress):
    nb_samples = len(segments)
    nb_validation_samples = int(np.ceil(nb_samples * percentage_validation_sampels))

    valid_class_path = os.path.join(source_dir, "valid", c)
    if not os.path.exists(valid_class_path):
        #print("os.makedirs("+valid_class_path+")")
        os.makedirs(valid_class_path)

    i_valid_samples = np.random.choice(range(len(segments)),
                                       nb_validation_samples, replace=False)
    valid_samples = [segments[i] for i in i_valid_samples]

    for sample in valid_samples:
        #print(c, "validation")
        for segment in sample:
            #print("shutil.move("+segment+","+valid_class_path+")")
            shutil.move(segment, valid_class_path)
