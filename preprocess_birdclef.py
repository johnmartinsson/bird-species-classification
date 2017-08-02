import numpy as np
np.random.seed(1337)

import xml.etree.ElementTree as ET
import glob
import csv
import os
import shutil
import tqdm
from bird import preprocessing as pp

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--xml_dir", dest="xml_dir")
parser.add_option("--wav_dir", dest="wav_dir")
parser.add_option("--is_testset", dest="is_testset")
parser.add_option("--output_dir", dest="output_dir")
(options, args) = parser.parse_args()

# Settings
segment_size_seconds = 3

# Paths
xml_paths = glob.glob(os.path.join(options.xml_dir, "*.xml"))
source_dir = options.wav_dir
preprocessed_dir = options.output_dir
noise_dir = os.path.join(preprocessed_dir, "noise")

print("Loading xml roots... ")
progress = tqdm.tqdm(range(len(xml_paths)))
xml_roots = [ET.parse(f) for (p, f) in zip(progress, xml_paths)]

if not os.path.exists(preprocessed_dir):
    print("Create diractory: ", preprocessed_dir)
    os.makedirs(preprocessed_dir)

if not os.path.exists(noise_dir):
    print("Create diractory: ", noise_dir)
    os.makedirs(noise_dir)

print("Preprocessing random species subset...")
progress = tqdm.tqdm(range(len(xml_roots)))
for (p, r) in zip(progress, xml_roots):
    species = ""
    if not options.is_testset:
        species = r.find("ClassId").text
    filename = r.find("FileName").text
    filepath = os.path.join(source_dir, filename)

    class_dir = os.path.join(preprocessed_dir, "signal", species)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # preprocess the sound file, and save signal to class_dir, noise to
    # noise_dir with specified segment size
    # print("Preprocess: {}".format(filepath))
    pp.preprocess_sound_file(filepath, class_dir, noise_dir,
                             segment_size_seconds)
