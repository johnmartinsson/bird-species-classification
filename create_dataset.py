import numpy as np
np.random.seed(1337)

import os
import glob
import shutil
import optparse
import tqdm

# src_dir must be organized as:
# src_dir:
#   class_1:
#     .
#     .
#     .
#   class_n:
#
# The dst_dir must NOT exist
parser = optparse.OptionParser()
parser.add_option("--src_dir", dest="src_dir")
parser.add_option("--dst_dir", dest="dst_dir")
parser.add_option("--subset_size", dest="subset_size")
parser.add_option("--valid_percentage", dest="valid_percentage")
(options, args) = parser.parse_args()

sound_classes = None

def choose_random_subset_classes(directory):
    sound_classes = os.listdir(directory)
    subset_size = int(options.subset_size)
    subset_sound_classes = np.random.choice(sound_classes, subset_size,
                                            replace=False)
    return subset_sound_classes

def choose_all_classes(directory):
    sound_classes = os.listdir(directory)
    return sound_classes

def get_basename_without_segment_id(fname):
    xs = fname.split("_")
    xs = xs[:len(xs)-2]
    return "_".join(xs)

def group_by_basename_without_segmend_id(fnames):
    groups = {}
    for fname in fnames:
        base_fname = get_basename_without_segment_id(fname)
        if base_fname not in groups:
            groups[base_fname] = [fname]
        else:
            groups[base_fname].append(fname)
    return list(groups.values())

def main():
    sound_classes = None
    src_dir = options.src_dir
    dst_dir = options.dst_dir
    valid_percentage = float(options.valid_percentage)

    if options.subset_size:
        sound_classes = choose_random_subset_classes(src_dir)
    else:
        sound_classes = choose_all_classes(src_dir)
    # print("sound classes:", sound_classes)

    dst_train_dir = os.path.join(dst_dir, "train")
    dst_valid_dir = os.path.join(dst_dir, "valid")

    os.makedirs(dst_dir)
    os.makedirs(dst_train_dir)
    os.makedirs(dst_valid_dir)
    print("copying sound classes...")
    for sound_class in sound_classes:
        sound_class_src_dir = os.path.join(src_dir, sound_class)
        sound_class_dst_dir = os.path.join(dst_train_dir, sound_class)
        shutil.copytree(sound_class_src_dir, sound_class_dst_dir)
    print("splitting train/validation...")
    for sound_class in sound_classes:
        sound_class_train_dir = os.path.join(dst_train_dir, sound_class)
        sound_class_valid_dir = os.path.join(dst_valid_dir, sound_class)
        if not os.path.exists(sound_class_valid_dir):
            os.makedirs(sound_class_valid_dir)
        sound_class_segments = glob.glob(os.path.join(sound_class_train_dir, "*.wav"))
        sound_class_segmentss = group_by_basename_without_segmend_id(sound_class_segments)
        nb_samples = len(sound_class_segmentss)
        nb_validation_samples = int(np.ceil(valid_percentage * nb_samples))
        nb_train_samples = nb_samples - nb_validation_samples
        validation_sampless = np.random.choice(sound_class_segmentss,
                                              nb_validation_samples,
                                              replace=False)
        for validation_samples in validation_sampless:
            for validation_sample in validation_samples:
                validation_dst_sample = os.path.join(sound_class_valid_dir,
                                                     os.path.basename(validation_sample))
                shutil.move(validation_sample, validation_dst_sample)

if __name__ == "__main__":
    main()
