from matplotlib import pyplot as plt
from matplotlib import gridspec

import configparser
import os
import pickle
import glob
import tqdm
import numpy as np
from skimage import morphology

from bird import loader
from bird import utils
from bird import preprocessing as pp
from bird import signal_processing as sp
from bird import data_augmentation as da
from bird import analysis as a
import data_analysis

def chunks(l, n):
    chunk_size = int(np.ceil(len(l)/n))
    """Yield n chunks from l."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]

def plot_confusion_matrix(picke_file, directory):
    with open(picke_file, 'rb') as input:
        stats = pickle.load(input)

    index_to_species = loader.build_class_index(directory)
    species_to_index = {v: k for k, v in index_to_species.items()}

    size = len(index_to_species.keys())
    confusion_matrix = np.zeros((size, size))
    for index, species in index_to_species.items():
        row = np.zeros(size)
        stat = stats[species]
        confusion = stat['confusion']
        for species, value in confusion.items():
            i = species_to_index[species]
            row[i] = value
        if np.sum(row) <= 0:
            print(species)
        confusion_matrix[index,:] = row

    for i in range(size):
        species = index_to_species[i]
        print(species, "(" + str(i) + ")", np.sum(confusion_matrix[:,i]))

    print("Total predictions:", np.sum(confusion_matrix))
    confusion_matrix = np.flip(confusion_matrix, 0)
    for i in range(size):
        if np.sum(confusion_matrix[i]) <= 0:
            print("Something is wrong")
        confusion_matrix[i] = confusion_matrix[i]/np.sum(confusion_matrix[i])


    cmap = plt.cm.get_cmap('jet')
    fig = plt.figure()
    # plt.gca().invert_yaxis()
    plt.pcolormesh(confusion_matrix, cmap=cmap)
    fig.savefig("confusion.png")

    fig.clf()
    plt.close(fig)

def plotaccuracy_by_trainingsamples(picke_file):
    with open(picke_file, 'rb') as input:
        stats = pickle.load(input)

    xs = zip(stats.keys(), stats.values())
    ys = [a for a in xs]
    xs = sorted(ys, key=lambda t: t[1]["training_samples"], reverse=True)
    xss = chunks(xs, 10)
    yss = []
    for xs in xss:
        ys = []
        for (species, x) in xs:
            correct = x["correct"]
            incorrect = x["incorrect"]
            accuracy = correct/(correct+incorrect)
            ys.append(accuracy)
        yss.append(ys)

    xs = []
    for ys in yss:
        xs.append(np.mean(ys))

    fig = plt.figure(1)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(xs, 'o-')
    plt.ylabel('accuracy')
    plt.xlabel('10% chunks of species')
    fig.savefig("accuracy_by_training_samples.png")

    fig.clf()
    plt.close(fig)

def plotaccuracy_by_trainingsegments(picke_file, directory):
    with open(picke_file, 'rb') as input:
        stats = pickle.load(input)

    def accuracy(r):
        return r[1]["correct"]/(r[1]["correct"]+r[1]["incorrect"])

    xs = zip(stats.keys(), stats.values())
    ys = [a for a in xs]
    xs = sorted(ys, key=lambda t: accuracy(t), reverse=True)


    segments = []
    accuracies = []
    for species, results in xs:
        segments.append(len(glob.glob(os.path.join(directory, "train", species,
                                                   "*.wav"))))
        accuracies.append(accuracy((species, results)))

    mean_segmentss = list(map(np.mean, chunks(segments, 10)))
    median_segmentss = list(map(np.median, chunks(segments, 10)))
    accuraciess = list(map(np.mean, chunks(accuracies, 10)))

    fig, ax1 = plt.subplots()
    ax1.plot(accuraciess, 'ro-')
    ax1.set_xlabel('sound class sorted by accuracy (descending)')
    ax1.set_ylabel('accuracy')

    ax2 = ax1.twinx()
    ax2.plot(mean_segmentss, 'bo-')
    ax2.plot(median_segmentss, 'go-')
    ax2.set_ylabel('segments')
    fig.savefig("accuracy_and_segments.png")

    fig.clf()
    plt.close(fig)

def plotaccuracy_by_validationsamples(picke_file):
    with open(picke_file, 'rb') as input:
        stats = pickle.load(input)

    xs = zip(stats.keys(), stats.values())
    ys = [a for a in xs]
    xs = sorted(ys, key=lambda t: t[1]["validation_samples"], reverse=True)
    xss = chunks(xs, 10)
    yss = []
    for xs in xss:
        ys = []
        for (species, x) in xs:
            correct = x["correct"]
            incorrect = x["incorrect"]
            accuracy = correct/(correct+incorrect)
            ys.append(accuracy)
        yss.append(ys)

    xs = []
    for ys in yss:
        xs.append(np.mean(ys))

    fig = plt.figure(1)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(xs, 'o-')
    plt.ylabel('accuracy')
    plt.xlabel('10% chunks of species')
    fig.savefig("accuracy_by_validation_samples.png")

    fig.clf()
    plt.close(fig)

def plot_accuracy_by_background_species(pickle_file, xml_roots):
    with open(pickle_file, 'rb') as input:
        stats = pickle.load(input)

    def accuracy(r):
        return r[1]["correct"]/(r[1]["correct"]+r[1]["incorrect"])

    xs = zip(stats.keys(), stats.values())
    ys = [a for a in xs]
    xs_sorted = sorted(ys, key=lambda t: accuracy(t), reverse=True)

    def parse_background_species(bs):
        if bs:
            return bs.split(",")
        else:
            return []

    # def union(ls1, ls2):
        # for l in ls2:
            # if l not in ls1:
                # ls1.append(l)
        # return ls1


    data = [{
        "class_id":r.find("ClassId").text
        , "background_species":parse_background_species(r.find("BackgroundSpecies").text)
    } for r in xml_roots]

    groups = data_analysis.groupby(data, lambda x: x["class_id"])

    ys = []
    xs = []
    for class_id, results in xs_sorted:
        ys.append(accuracy((class_id, results)))
        for key, group in groups:
            if key == class_id:
                background_species = []
                for value in group:
                    for bs in value["background_species"]:
                        background_species.append(bs)
                print(class_id, len(set(background_species)))
                xs.append(len(set(background_species)))

    fig, ax1 = plt.subplots()
    ax1.plot(ys, 'ro-')
    ax1.set_xlabel('sound class sorted by accuracy (descending)')
    ax1.set_ylabel('accuracy')

    ax2 = ax1.twinx()
    ax2.plot(xs, 'bo-')
    ax2.set_ylabel('background species')
    fig.savefig("accuracy_and_background_species.png")

    fig.clf()
    plt.close(fig)




def plot_sound_class_sorted_by_accuracy(pickle_file):
    with open(pickle_file, 'rb') as input:
        stats = pickle.load(input)

    def accuracy(stat):
        return stat[1]["correct"]/(stat[1]["correct"]+stat[1]["incorrect"])

    xs = zip(stats.keys(), stats.values())
    ys = [a for a in xs]
    xs_sorted = sorted(ys, key=lambda t: accuracy(t), reverse=True)
    # xs = [accuracy(t) for t in xs_sorted]
    # xss = chunks(xs_sorted, 20)

    # yss = []
    # for xs in xss:
    ys = []
    for stat in xs_sorted:
        ys.append(accuracy(stat))
        # yss.append(ys)
    # xs = []
    # for ys in yss:
        # xs.append(np.mean(ys))

    fig = plt.figure(1)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(ys, '-')
    plt.ylabel('accuracy')
    plt.xlabel('sound class sorted by accuracy (descending)')
    fig.savefig("sound_class_sorted_by_accuracy.png")

    fig.clf()
    plt.close(fig)

def create_top_bot_table(pickle_file, size):
    with open(pickle_file, 'rb') as input:
        stats = pickle.load(input)

    def accuracy(stat):
        return stat[1]["correct"]/(stat[1]["correct"]+stat[1]["incorrect"])

    xs = zip(stats.keys(), stats.values())
    ys = [a for a in xs]
    xs = sorted(ys, key=lambda t: accuracy(t), reverse=True)
    print("total samples", sum([stat[1]["training_samples"] for stat in xs]))
    top = xs[:size]
    bot = xs[len(xs)-size:]

    print("TOP " + str(size))
    for stat in top:
        # print(stat)
        print(stat[0], "accuracy:", accuracy(stat), "training_samples:",
              stat[1]["training_samples"])
    print("samples", sum([stat[1]["training_samples"] for stat in top]))
    print("")

    print("BOT " + str(size))
    for stat in bot:
        # print(stat)
        print(stat[0], "accuracy:", accuracy(stat), "training_samples:",
              stat[1]["training_samples"])
    print("samples", sum([stat[1]["training_samples"] for stat in bot]))

    return [stat[0] for stat in top], [stat[0] for stat in bot]

def plot_accuracy_and_structure(pickle_file, directory):
    accuracies = []
    valid_structures = []
    train_structures = []
    with open(pickle_file, 'rb') as input:
        stats = pickle.load(input)

    def accuracy(stat):
        return stat[1]["correct"]/(stat[1]["correct"]+stat[1]["incorrect"])

    key_value_tuples = zip(stats.keys(), stats.values())
    key_value_tuples_sorted = sorted(key_value_tuples, key=lambda t: accuracy(t), reverse=True)

    for (species, values) in key_value_tuples_sorted:
        train_class_path = os.path.join(directory, "train", species)
        valid_class_path = os.path.join(directory, "valid", species)
        train_structure = a.compute_class_structure(train_class_path)
        valid_structure = a.compute_class_structure(valid_class_path)
        accuracy = values["correct"]/(values["correct"]+values["incorrect"])
        accuracies.append(accuracy)
        valid_structures.append(valid_structure/values["validation_samples"])
        train_structures.append(train_structure/values["training_samples"])

    accuraciess = list(map(np.mean, chunks(accuracies, 10)))
    valid_structuress = list(map(np.mean, chunks(valid_structures, 10)))
    train_structuress = list(map(np.mean, chunks(train_structures, 10)))

    fig, ax1 = plt.subplots()
    ax1.plot(accuraciess, 'ro-')
    ax1.set_xlabel('sound class sorted by accuracy (descending)')
    ax1.set_ylabel('accuracy')

    ax2 = ax1.twinx()
    ax2.plot(valid_structuress, 'bo-')
    ax2.plot(train_structuress, 'go-')
    ax2.set_ylabel('structure')
    fig.savefig("accuracy_and_structure.png")

    fig.clf()
    plt.close(fig)



def plot_accuracy_and_energy(pickle_file, directory):
    accuracies = []
    valid_energies = []
    train_energies = []
    with open(pickle_file, 'rb') as input:
        stats = pickle.load(input)

    def accuracy(stat):
        return stat[1]["correct"]/(stat[1]["correct"]+stat[1]["incorrect"])

    key_value_tuples = zip(stats.keys(), stats.values())
    key_value_tuples_sorted = sorted(key_value_tuples, key=lambda t: accuracy(t), reverse=True)

    for (species, values) in key_value_tuples_sorted:
        train_class_path = os.path.join(directory, "train", species)
        valid_class_path = os.path.join(directory, "valid", species)
        train_energy = a.compute_class_energy(train_class_path)
        valid_energy = a.compute_class_energy(valid_class_path)
        accuracy = values["correct"]/(values["correct"]+values["incorrect"])
        accuracies.append(accuracy)
        valid_energies.append(valid_energy/values["validation_samples"])
        train_energies.append(train_energy/values["training_samples"])

    accuraciess = list(map(np.mean, chunks(accuracies, 10)))
    valid_energiess = list(map(np.mean, chunks(valid_energies, 10)))
    train_energiess = list(map(np.mean, chunks(train_energies, 10)))

    fig, ax1 = plt.subplots()
    ax1.plot(accuraciess, 'ro-')
    ax1.set_xlabel('sound class sorted by accuracy (descending)')
    ax1.set_ylabel('accuracy')

    ax2 = ax1.twinx()
    ax2.plot(valid_energiess, 'bo-')
    ax2.plot(train_energiess, 'go-')
    ax2.set_ylabel('energy')
    fig.savefig("accuracy_and_energy.png")

    fig.clf()
    plt.close(fig)

def compute_and_save_spectrograms_for_files(files):
    progress = tqdm.tqdm(range(len(files)))
    for (f, p) in zip(files, progress):
        img_log_spectrogram_from_wave_file(f)
        img_spectrogram_from_wave_file(f)

def img_log_spectrogram_from_wave_file(filepath):
    fs, x = utils.read_gzip_wave_file(filepath)
    Sxx = sp.wave_to_amplitude_spectrogram(x, fs, 512, 128)[:256]
    baseName = utils.get_basename_without_ext(filepath)
    save_matrix_to_file(Sxx, "Amplitude Spectrogram", baseName + "_LOG.png")

def img_spectrogram_from_wave_file(filepath):
    fs, x = utils.read_gzip_wave_file(filepath)
    Sxx = sp.wave_to_log_amplitude_spectrogram(x, fs, 512, 128)[:256]
    baseName = utils.get_basename_without_ext(filepath)
    save_matrix_to_file(Sxx, "Log Amplitude Spectrogram", baseName + "_AMP.png")

def sprengel_binary_mask_from_wave_file(filepath):
    fs, x = utils.read_wave_file(filepath)
    Sxx = sp.wave_to_amplitude_spectrogram(x, fs)
    Sxx_log = sp.wave_to_log_amplitude_spectrogram(x, fs)

    # plot spectrogram
    fig = plt.figure(1)
    subplot_image(Sxx_log, 411, "Spectrogram")

    Sxx = pp.normalize(Sxx)
    binary_image = pp.median_clipping(Sxx, 3.0)

    subplot_image(binary_image + 0, 412, "Median Clipping")

    binary_image = morphology.binary_erosion(binary_image, selem=np.ones((4, 4)))

    subplot_image(binary_image + 0, 413, "Erosion")

    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    subplot_image(binary_image + 0, 414, "Dilation")

    mask = np.array([np.max(col) for col in binary_image.T])
    mask = morphology.binary_dilation(mask, np.ones(4))
    mask = morphology.binary_dilation(mask, np.ones(4))

    # plot_vector(mask, "Mask")

    fig.set_size_inches(10, 12)
    plt.tight_layout()
    fig.savefig(utils.get_basename_without_ext(filepath) + "_binary_mask.png", dpi=100)

def signal_and_noise_spectrogram_from_wave_file(filepath):

    (fs, wave) = utils.read_wave_file(filepath)
    spectrogram = sp.wave_to_sample_spectrogram(wave, fs)
    signal_wave, noise_wave = pp.preprocess_wave(wave, fs)
    spectrogram_signal = sp.wave_to_sample_spectrogram(signal_wave, fs)
    spectrogram_noise = sp.wave_to_sample_spectrogram(noise_wave, fs)

    fig = plt.figure(1)
    cmap = plt.cm.get_cmap('jet')
    gs = gridspec.GridSpec(2, 2)
    # whole spectrogram
    ax1 = fig.add_subplot(gs[0,:])
    ax1.pcolormesh(spectrogram, cmap=cmap)
    ax1.set_title("Sound")

    ax2 = fig.add_subplot(gs[1,0])
    ax2.pcolormesh(spectrogram_signal, cmap=cmap)
    ax2.set_title("Signal")

    ax3 = fig.add_subplot(gs[1,1])
    ax3.pcolormesh(spectrogram_noise, cmap=cmap)
    ax3.set_title("Noise")

    gs.update(wspace=0.5, hspace=0.5)

    basename = utils.get_basename_without_ext(filepath)
    fig.savefig(basename+"_noise_signal.png")

    fig.clf()
    plt.close(fig)

def same_class_augmentation_from_dir(class_dir):
    sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    sig_path = np.random.choice(sig_paths, 1, replace=False)[0]
    (fs, sig) = utils.read_wave_file(sig_path)

    aug_sig_path = np.random.choice(sig_paths, 1, replace=False)[0]
    (fs, aug_sig) = utils.read_wave_file(aug_sig_path)
    alpha = np.random.rand()
    combined_sig = (1.0-alpha)*sig + alpha*aug_sig

    spectrogram_sig = sp.wave_to_sample_spectrogram(sig, fs)
    spectrogram_aug_sig = sp.wave_to_sample_spectrogram(aug_sig, fs)
    spectrogram_combined_sig = sp.wave_to_sample_spectrogram(combined_sig, fs)

    fig = plt.figure(1)
    cmap = plt.cm.get_cmap('jet')
    gs = gridspec.GridSpec(3, 1)
    # whole spectrogram
    ax1 = fig.add_subplot(gs[0,0])
    ax1.pcolormesh(spectrogram_sig, cmap=cmap)
    ax1.set_title("Signal 1")

    ax2 = fig.add_subplot(gs[1,0])
    ax2.pcolormesh(spectrogram_aug_sig, cmap=cmap)
    ax2.set_title("Signal 2")

    ax3 = fig.add_subplot(gs[2,0])
    ax3.pcolormesh(spectrogram_combined_sig, cmap=cmap)
    ax3.set_title("Augmented Signal (alpha=" + str(alpha) + ")")

    gs.update(wspace=0.5, hspace=0.5)

    basename = utils.get_basename_without_ext(sig_path)
    fig.savefig(basename+"_same_class_augmentation.png")

    fig.clf()
    plt.close(fig)


def noise_augmentation_from_dirs(noise_dir, class_dir):
    sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    sig_path = np.random.choice(sig_paths, 1, replace=False)[0]
    (fs, sig) = utils.read_wave_file(sig_path)
    aug_sig = da.noise_augmentation(sig, noise_dir)

    spectrogram_sig = sp.wave_to_sample_spectrogram(sig, fs)
    spectrogram_aug_sig = sp.wave_to_sample_spectrogram(aug_sig, fs)

    fig = plt.figure(1)
    cmap = plt.cm.get_cmap('jet')
    gs = gridspec.GridSpec(2, 1)
    # whole spectrogram
    ax1 = fig.add_subplot(gs[0,0])
    ax1.pcolormesh(spectrogram_sig, cmap=cmap)
    ax1.set_title("Original Signal")

    ax2 = fig.add_subplot(gs[1,0])
    ax2.pcolormesh(spectrogram_aug_sig, cmap=cmap)
    ax2.set_title("Noise Augmented signal")

    gs.update(wspace=0.5, hspace=0.5)

    basename = utils.get_basename_without_ext(sig_path)
    fig.savefig(basename+"_noise_augmentation.png")

    fig.clf()
    plt.close(fig)



def save_matrix_to_file(Sxx, title, filename):
    cmap = plt.cm.get_cmap('jet')
    #cmap = grayify_cmap('cubehelix_r')
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Window Samples')
    fig.savefig(filename)
    # close
    plt.clf()
    plt.close()

def plot_history_to_image_file(pickle_path):
    with open(pickle_path, 'rb') as input:
        trainLoss = pickle.load(input)
        validLoss = pickle.load(input)
        trainAcc = pickle.load(input)
        validAcc = pickle.load(input)

        x = range(len(validAcc))
        y = validAcc
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)

        fig = plt.figure(1)
        plt.subplot(211)
        axes = plt.gca()
        axes.set_ylim([0, 10])
        plt.ylabel("Loss")
        plt.plot(trainLoss, 'o-', label="train")
        plt.plot(validLoss, 'o-', label="valid")
        plt.legend(loc="upper_left")
        plt.subplot(212)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.plot(trainAcc, 'o-', label="train")
        plt.plot(validAcc, 'o-', label="valid")
        # plt.plot(p(x), 'o-', label="trend")
        plt.legend(loc="upper_left")

        basename = utils.get_basename_without_ext(pickle_path)
        fig.savefig(basename + ".png")
        plt.clf()
        plt.close()

def plot_histories_to_image_file(directories):
    config_parser = configparser.ConfigParser()
    fig = plt.figure(1)
    for d in directories:
        config_parser.read(os.path.join(d, "conf.ini"))
        optimizer = config_parser['TRAINING']['Optimizer']
        history_path = os.path.join(d, "history.pkl")
        with open(history_path, "rb") as input:
            trainLoss = pickle.load(input)
            validLoss = pickle.load(input)
            trainAcc = pickle.load(input)
            validAcc = pickle.load(input)
            plt.subplot(211)
            axes = plt.gca()
            axes.set_ylim([0, 20])
            plt.ylabel("Validation Loss")
            # plt.plot(trainLoss, '-', label="train")
            plt.plot(validLoss, '-', label=optimizer)
            plt.legend(loc="upper_left")
            plt.subplot(212)
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.ylabel("Validation Accuracy")
            plt.xlabel("Epoch")
            # plt.plot(trainAcc, 'o-', label="train")
            plt.plot(validAcc, '-', label=optimizer)
            # plt.plot(p(x), 'o-', label="trend")
            # plt.legend(loc="upper_left")

    fig.savefig("optimizer_history.png")
    plt.clf()
    plt.close()




def plot_log_spectrogram_from_wave_file(filename):
    fs, x = utils.read_gzip_wave_file(filename)
    Sxx = sp.wave_to_log_amplitude_spectrogram(x, fs, 512, 128)[:256]
    Sxx2 = utils.wave_to_log_spectrogram_aux(x, fs)
    plot_matrix(Sxx, "Log Amplitude Spectrogram")
    plot_matrix(Sxx2, "Log Amplitude Spectrogram (scipy)")

def plot_spectrogram_from_wave_file(filename):
    fs, x = utils.read_gzip_wave_file(filename)
    Sxx = sp.wave_to_amplitude_spectrogram(x, fs, 512, 128)[:256]
    plot_matrix(Sxx, "Amplitude Spectrogram")

def subplot_image(Sxx, n_subplot, title):
    cmap = grayify_cmap('cubehelix_r')
    # cmap = plt.cm.get_cmap('jet')
    # cmap = grayify_cmap('jet')
    plt.subplot(n_subplot)
    plt.title(title)
    plt.pcolormesh(Sxx, cmap=cmap)

def plot_matrix(Sxx, title):
    # cmap = grayify_cmap('cubehelix_r')
    cmap = plt.cm.get_cmap('jet')
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    plt.pcolormesh(Sxx, cmap=cmap)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Window Samples')
    plt.show()

def plot_vector(x, title):
    mesh = np.zeros((257, x.shape[0]))
    for i in range(x.shape[0]):
        mesh[256][i] = x[i] * 2500
        mesh[255][i] = x[i] * 2500
        mesh[254][i] = x[i] * 2500
    plot_matrix(mesh, title)

def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)
