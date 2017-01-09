import os
import time
import pickle
from time import localtime, strftime
from subprocess import call

# Training Settings
nb_iterations = 5

model_name = "cuberun"
train_path = "datasets/birdClef2016Subset/train"
valid_path = "datasets/birdClef2016Subset/valid"
lock_file  = "job_done.lock"

basename = strftime("%Y_%m_%d_%H:%M:%S_", localtime()) + model_name
weight_file_path = os.path.join("./weights", basename + ".h5")
history_file_path = os.path.join("./history", basename + ".pkl")
tmp_history_file_path = os.path.join("./history", basename + "_tmp.pkl")


# Arguments
args = [
    "--weight_path=" + weight_file_path,
    "--history_path=" + tmp_history_file_path,
    "--train_path=" + train_path,
    "--valid_path=" + valid_path,
    "--lock_file=" + lock_file
]


def train():
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    for i in range(nb_iterations):

        # create lock file
        open(lock_file, 'a').close()
        # submit job, train once
        if not i == 0:
            call(["python", "train_model.py"] + args +
                 ["--first_epoch="+str(False)])
        else:
            call(["python", "train_model.py"] + args +
                 ["--first_epoch="+str(True)])
        # block until job is finished
        while os.path.exists(lock_file):
            time.sleep(1)

        print("iteration: " + str(i) + " is done.")

        # load all history data and append
        with open(tmp_history_file_path, 'rb') as input:
            train_loss = train_loss + pickle.load(input)
            valid_loss = valid_loss + pickle.load(input)
            train_acc = train_acc + pickle.load(input)
            valid_acc = valid_acc + pickle.load(input)

        # save all collected history data
        with open(history_file_path, 'wb') as output:
            pickle.dump(train_loss, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_loss, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_acc, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_acc, output, pickle.HIGHEST_PROTOCOL)
