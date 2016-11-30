import numpy as np
np.random.seed(42)

import xml.etree.ElementTree as ET
import glob
import csv
import os

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

species2id = {}
id_counter = 0

total_recodrings_subset = 0
for k in random_species:
    if not k in species2id:
        species2id[k] = id_counter
        id_counter += 1

    print(k, ":", species_count[k])
    total_recodrings_subset += species_count[k]

print("Total recordings subset:", total_recodrings_subset)

subset = []

species2wavpaths = {}
for r in xml_roots:
    species = r.find("Species").text
    wavbasename = os.path.splitext(r.find("FileName").text)[0]
    if species in random_species:
        if not species in species2wavpaths:
            species2wavpaths[species] = [wavbasename]
        else:
            species2wavpaths[species].append(wavbasename)

with open("train_paths.txt", 'w') as train_paths_file:
    with open("valid_paths.txt", 'w') as valid_paths_file:
        for (species, paths) in species2wavpaths.items():
            nb_paths = len(paths)
            nb_validation = int(1/3.0 * nb_paths)
            np.random.shuffle(paths)
            validation_paths = paths[:nb_validation]
            train_paths = paths[nb_validation:]
            for p in validation_paths:
                # create validation set
                valid_paths_file.write(p + "\n")
            for p in train_paths:
                # create train set
                train_paths_file.write(p + "\n")

with open("file2labels.csv", 'w') as file2labels:
    with open("subset_paths.txt", 'w') as subset_paths:
        file2labelswriter = csv.writer(file2labels)
        for r in xml_roots:
            species = r.find("Species").text
            if species in random_species:
                file2labelswriter.writerow([os.path.splitext(r.find("FileName").text)[0], species2id[species]])
                subset_paths.write(r.find("FileName").text + ".gz\n")

print("Subset size:", len(subset))
print("Species2Id:", species2id)
