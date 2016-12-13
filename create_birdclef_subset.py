import numpy as np
np.random.seed(42)

import xml.etree.ElementTree as ET
import glob
import csv
import os
import shutil
from bird import preprocessing as pp

xml_paths = glob.glob("datasets/birdClef2016/xml/*.xml")
xml_roots = [ET.parse(f) for f in xml_paths]
root2path = {}

species_count = {}

for r in xml_roots:
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
random_species = np.random.choice(species, 20)
xml_random_species = []

for r in xml_roots:
    species = r.find("Species").text
    if species in random_species:
        xml_random_species.append(r)

source_dir = "./datasets/birdClef2016/wav"
subset_root_dir = "./datasets/birdClefSubset"
noise_dir = os.path.join(subset_root_dir, "noise")

if not os.path.exists(subset_root_dir):
    os.makedirs(subset_root_dir)

if not os.path.exists(noise_dir):
    os.makedirs(noise_dir)

for r in xml_random_species:
    species = r.find("Species").text
    filename = r.find("FileName").text
    filepath = os.path.join(source_dir, filename)

    class_dir = os.path.join(subset_root_dir, species)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    pp.preprocess_sound_file(filename, class_dir, noise_dir, 3)
