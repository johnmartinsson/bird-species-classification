import os
import time
import pickle
from time import localtime, strftime
from subprocess import call

# Training Settings
nb_iterations = 20

model_name = "cuberun"
train_path = "/disk/martinsson-spring17/birdClef2016Whole/train"
valid_path = "/disk/martinsson-spring17/birdClef2016Whole/valid"
noise_path = "/home/martinsson-spring17/data/noise"
lock_file  = "job_done.lock"

basename = strftime("%Y_%m_%d_%H:%M:%S_", localtime()) + model_name
weight_file_path = os.path.join("./weights", basename + ".h5")
history_file_path = os.path.join("./history", basename + ".pkl")
tmp_history_file_path = os.path.join("./history", basename + "_tmp.pkl")

# Arguments
qsub_args = [
    "-cwd",
    "-l", "gpu=1",
    "-e", "./log/run_job.sh.error",
    "-o", "./log/run_job.sh.log",
    "./run_job.sh",
    weight_file_path,
    tmp_history_file_path,
    train_path,
    valid_path,
    noise_path,
    lock_file
]

def train():
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    for i in range(nb_iterations):

        # create lock file
        print("Creating lock file: ", lock_file)
        open(lock_file, 'a').close()

        # submit job, train once
        print("Submitting job number: ", str(i))
        if not i == 0:
            call(["qsub"] + qsub_args + ['False'])
        else:
            call(["qsub"] + qsub_args + ['True'])

        # block until job is finished
        while os.path.exists(lock_file):
            time.sleep(5)

        print("Job: " + str(i) + " is done.")

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
