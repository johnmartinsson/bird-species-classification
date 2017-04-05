import numpy as np
np.random.seed(42)

import xml.etree.ElementTree as ET
import glob
import csv
import os
import shutil
import tqdm
from bird import preprocessing as pp

xml_paths = glob.glob("datasets/birdClef2016/xml/*.xml")

print("Loading xml roots... ")
progress = tqdm.tqdm(range(len(xml_paths)))
xml_roots = [ET.parse(f) for (p, f) in zip(progress, xml_paths)]
root2path = {}

species_count = {}

print("Couting species...")
progress = tqdm.tqdm(range(len(xml_roots)))
for (p, r) in zip(progress, xml_roots):
    species = r.find("Species").text
    if species in species_count:
        species_count[species] += 1
    else:
        species_count[species] = 1

total_recordings = 0
min_count = 10000
max_count = 0
species = []
for k in species_count.keys():
    print(k, ":", species_count[k])
    total_recordings += species_count[k]
    min_count = min(species_count[k], min_count)
    max_count = max(species_count[k], max_count)
    species.append(k)

print("")
print("--------------------------------------")
print("Meta information")
print("--------------------------------------")
print("Number of species:", len(species_count.keys()))
print("Max:", max_count)
print("Min:", min_count)

assert(total_recordings == len(xml_paths))
print("--------------------------------------------------------------------------------")
print("- Chosing random species")
print("--------------------------------------------------------------------------------")

# Settings
nb_species = 20
segment_size_seconds = 3

# Paths
source_dir = "./datasets/birdClef2016/wav"
subset_root_dir = "./datasets/birdClef2016Whole"
noise_dir = os.path.join(subset_root_dir, "noise")

# Begin preprocessing
#random_species = np.random.choice(species, nb_species)
random_species = species

xml_random_species = []
print("Extracting random subset of xml_roots...")
progress = tqdm.tqdm(range(len(xml_roots)))
for (p, r) in zip(progress, xml_roots):
    species = r.find("Species").text
    if species in random_species:
        xml_random_species.append(r)


if not os.path.exists(subset_root_dir):
    print("Create diractory: ", subset_root_dir)
    os.makedirs(subset_root_dir)

if not os.path.exists(noise_dir):
    print("Create diractory: ", noise_dir)
    os.makedirs(noise_dir)

print("Preprocessing random species supset...")
progress = tqdm.tqdm(range(len(xml_random_species)))
for (p, r) in zip(progress, xml_random_species):
    species = r.find("Species").text
    filename = r.find("FileName").text
    filepath = os.path.join(source_dir, filename)

    r = np.random.rand()
    class_dir = os.path.join(subset_root_dir, "train", species)

    if not os.path.exists(class_dir):
        #print("Create diractory: ", class_dir)
        os.makedirs(class_dir)

    # preprocess the sound file, and save signal to class_dir, noise to
    # noise_dir with specified segment size
    pp.preprocess_sound_file(filepath, class_dir, noise_dir,
                             segment_size_seconds)
