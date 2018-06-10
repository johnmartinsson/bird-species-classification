#!/usr/bin/python3
import os
import time
import pickle
import configparser
import shutil
from time import localtime, strftime
from subprocess import call
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--config_file", dest="config_file")
(options, args) = parser.parse_args()

config_file = options.config_file
config_parser = configparser.ConfigParser()
config_parser.read(config_file)

model_name = config_parser['MODEL']['ModelName']
basename = config_parser['PATHS']['BaseName']
nb_iterations = int(config_parser['MODEL']['NumberOfIterations'])


if os.path.exists(basename):
    basename = basename
    config_file = os.path.join(basename, config_file)
else:
    basename = strftime("%Y_%m_%d_%H%M%S_", localtime()) + model_name
    os.makedirs(basename)
    # copy configuration file
    shutil.copyfile(config_file, os.path.join(basename, config_file))
    config_file = os.path.join(basename, config_file)

weight_file_path = os.path.join(basename, "weights.h5")
history_file_path = os.path.join(basename, "history.pkl")
tmp_history_file_path = os.path.join(basename, "history_tmp.pkl")
lock_file  = os.path.join(basename, "file.lock")

# This was designed to run in a queue system. If this is not what you want to do
# simply comment out the five first elements in the qsub_args list, and just run
# the script directly on the GPU/CPU.
qsub_args = [
    # "qsub",
    # "-cwd",
    # "-l", "gpu=1",
    # "-e", os.path.join(basename, "stderr.error"),
    # "-o", os.path.join(basename, "stdout.log"),
    "./run_job.sh",
    weight_file_path,
    tmp_history_file_path,
    lock_file,
    config_file
]

def train():
    print("#############################")
    print("# Training Settings")
    print("#############################")
    print("Model        : ", model_name)
    print("Weight path  : ", weight_file_path)
    print("History path : ", history_file_path)

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    # if exists means we are restarting a crashed training
    if os.path.isfile(history_file_path):
        print("Loading previous history data...")
        with open(history_file_path, 'rb') as input:
            train_loss = pickle.load(input)
            valid_loss = pickle.load(input)
            train_acc = pickle.load(input)
            valid_acc = pickle.load(input)

    for i in range(nb_iterations):

        # create lock file
        print("Creating lock file: ", lock_file)
        open(lock_file, 'a').close()

        # submit job, train once
        print("Submitting Job ", str(i), "/", str(nb_iterations))
        if not i == 0:
            call(qsub_args + ['False'])
        else:
            call(qsub_args + ['True'])

        # block until job is finished
        while os.path.exists(lock_file):
            time.sleep(5)

        print("Job " + str(i) + " is done.")

        # load all history data and append
        print("Loading temporary history data...")
        with open(tmp_history_file_path, 'rb') as input:
            train_loss = train_loss + pickle.load(input)
            valid_loss = valid_loss + pickle.load(input)
            train_acc = train_acc + pickle.load(input)
            valid_acc = valid_acc + pickle.load(input)

        # save all collected history data
        print("Save all collected history data...")
        with open(history_file_path, 'wb') as output:
            pickle.dump(train_loss, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_loss, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_acc, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_acc, output, pickle.HIGHEST_PROTOCOL)

train()
