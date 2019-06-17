# DISCLAIMER
This repository is not actively maintained. It is the result of a master's thesis and the code has been made available as a reference if anyone would like to reproduce the results of the thesis.

# Bird Species Classification
These are the project files for a master's thesis carried out at Chalmers University of Technology. The aim of the project was to improve upon a state-of-the-art bird species classifier by using deep residual neural networks, multiple-width frequency-delta data augmentation, and meta-data fusion to build and train a bird species classifier on bird song data with corresponding species labels.

- [Master's Thesis](http://publications.lib.chalmers.se/records/fulltext/249467/249467.pdf)
- [Presentation](http://johnmartinsson.github.io/thesis-presentation)

Please cite the master's thesis if this repository is useful for your research.

## Setup
```bash
$ git clone https://github.com/johnmartinsson/bird-species-classification
$ virtualenv -p /usr/bin/python3.6 venv
$ source venv/bin/activate
(venv)$ pip install -r requirements.txt

# Ubuntu/Linux 64-bit, CPU only, Python 3.6
(venv)$ pip install --upgrade tensorflow
# Ubuntu/Linux 64-bit, GPU enabled, Python 3.6
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Install from sources" below.
(venv)$ pip install --upgrade tensorflow-gpu
```

# Usage Instructions
This section explains how to preprocess the birdCLEF2016 dataset, how to split the data set in to a training and validation set, how to train a model on the training data, and how to evaluate the model on the validation data.

## Preprocess
First we need to down-sample the sound recordings.

```bash
$ # Resample to 22050 Hz (stand in wav directory)
$ for i in *; do sox $i -r 22050 tmp.wav; mv tmp.wav $i; done
```

Secondly, the signal parts, and the noise parts of the recordings are extracted and split into three second segments. The signal segments are put in different directories depending on the class given in the xml data, and all noise segments are put in a separate noise directory.
```bash
$ python preprocess_birdclef.py --xml_dir=<path-to-xml-dir> \
                                --wav_dir=<path-to-wav-dir> \
                                --output_dir=<path-to-output-dir>
```

Lastly, the data is split into a training set and a validation set:

```bash
$ python create_dataset.py --src_dir=<path-to-signal-dir> \
                           --dst_dir=<path-to-destination-dir> \
                           --subset_size=<subset-size> \
                           --valid_percentage=<validation-percentage>
```
where src points to the signal segments, dst is the destination, subset size is an optional argument which makes training and validation data a randomly chosen subset of bird species from the whole data set, and the valid percentage is how many percent of the data that should be in the validation set.

## Train
```bash
$ python train.py --config_file=conf.ini
```

## Run Predictions
```bash
$ python run_predictions.py --experiment_path=<path-to-experiment>
```

## Evaluation
```bash
$ python evaluate.py --experiment_path=<path-to-results>
```
# Models
In this project two different models have been used: a reimplementation of Elias Sprengels [winning solution](http://ceur-ws.org/Vol-1609/16090547.pdf) for the BirdCLEF 2016 challenge, and a Keras implementation of the [deep residual neural network](https://github.com/raghakot/keras-resnet/blob/master/resnet.py).

# Libraries
The following libraries are used in this method:

- [keras](http://keras.io/),
- [scipy](https://www.scipy.org/),
- [numpy](http://www.numpy.org/),
- [scikit-learn](http://scikit-learn.org/).

# Evaluation Methods
- [Mean Average Precision](https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py)
- [Coverage Error](http://scikit-learn.org/stable/modules/model_evaluation.html#coverage-error)
- [Label Ranking Average Precision](http://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision)
- [AUROC](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)

# Challenges
This is a collection of bird species classification challenges that, has been, and are carried out around the world.

## BirdCLEF: an audio record-based bird identification task
- [BirdCLEF 2016](http://www.imageclef.org/lifeclef/2016/bird)
- [BirdCLEF 2017](http://www.imageclef.org/lifeclef/2017/bird)

### Solutions and Source Code
- Rank 1 BirdCLEF 2016 [solution description](http://ceur-ws.org/Vol-1609/16090547.pdf)

## Bird Audio Detection Challenge
- [Bird Audio Detection Challenge](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/),
- [Survey Paper](https://arxiv.org/pdf/1608.03417v1.pdf) and [Discussion](https://groups.google.com/forum/#!forum/bird-detection),
- [Blog Article](http://machine-listening.eecs.qmul.ac.uk/2016/10/bird-audio-detection-baseline-generalisation/): Generalization in Bird Audio Detection.

## MLSP 2013 Bird Classification Challenge
- [MLSP 2013 Bird Classification Challenge](https://www.kaggle.com/c/mlsp-2013-birds/).

### Solutions and Source Code
Original compilation source: [xuewei4d](https://github.com/xuewei4d/KaggleSolutions)

- Rank 1 [solution code](https://github.com/gaborfodor/MLSP_2013) and [description](https://www.kaggle.com/c/mlsp-2013-birds/forums/t/5457/congratulations-to-the-winners/29159#post29159) by beluga,
- Rank 2 [solution description](https://www.kaggle.com/c/mlsp-2013-birds/forums/t/5457/congratulations-to-the-winners/29017#post29017) by Herbal Candy,
- Rank 3 [solution description](https://www.kaggle.com/c/mlsp-2013-birds/forums/t/5457/congratulations-to-the-winners/29101#post29101) by Anil Thomas,
- Rank 4 [solution description](http://www.kaggle.com/c/mlsp-2013-birds/forums/t/5457/congratulations-to-the-winners/29092#post29092) by Maxim Milakov,
- [Solution thread](https://www.kaggle.com/c/mlsp-2013-birds/forums/t/5457/congratulations-to-the-winners).

# Applications
This is a collection of applications which solve a similar problem.

## Warbler
- [Warbler](https://warblr.net/).
