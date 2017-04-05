#!/usr/bin/python
import os
import glob
import numpy as np
from optparse import OptionParser
from bird import loader

parser = OptionParser()
parser.add_option("--valid_path", dest="valid_path")
parser.add_option("--train_path", dest="train_path")
(options, args) = parser.parse_args()

validation = options.valid_path
train = options.train_path

stats = []

for species in os.listdir(train):
    nb_valid = len(loader.group_segments(glob.glob(os.path.join(validation, species, "*.wav"))))
    nb_train = len(loader.group_segments(glob.glob(os.path.join(train, species, "*.wav"))))
    stat = {
        "species":species,
        "validation_samples":nb_valid,
        "training_samples":nb_train
    }
    stats.append(stat)

stats_sorted = sorted(stats, key=lambda x: x['training_samples'],
                      reverse=True)
top_10 = stats_sorted[:10]
bot_10 = stats_sorted[len(stats_sorted)-10:]
print("Top 10")
[print(x['species'], x['training_samples']) for x in top_10]
print("Bot 10")
[print(x['species'], x['training_samples']) for x in bot_10]
print("proportion")
[print(x['species'], x['validation_samples']/(x['training_samples']+x['validation_samples'])) for x in stats_sorted]
