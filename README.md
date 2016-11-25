# Bird Species Classification
Using convolutional neural networks to build and train a bird species classifier on bird song data with corresponding species labels.

## Setup
```bash
$ git clone https://github.com/johnmartinsson/bird-species-classification
$ virtualenv -p /usr/bin/python3.5 venv
$ source venv/bin/activate
(venv)$ pip install -r requirements.txt

# Ubuntu/Linux 64-bit, CPU only, Python 3.5
(venv)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl
# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Install from sources" below.
(venv)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl

# Install tensorflow
(venv)$ pip3 install --upgrade $TF_BINARY_URL
```

# Usage Instriction
Note that these instructions can __not__ be followed right now, but they are rather here as a guidline of an interface that could be implemented.

## Train
The training, and validation data folders should contain the sound files, and a csv file which maps the name of a sound file to a set of ground truth labels.

## Test

## Predict

# Libraries
The following libraries are used in this method:

- [keras](http://keras.io/),
- [scipy](https://www.scipy.org/),
- [numpy](http://www.numpy.org/),
- [scikit-learn](http://scikit-learn.org/).

# Evaluation Methods
- [Mean Average Precision](https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py)

# Challenges
This is a collection of bird species classification challenges that, has been, and is carried out around the world.

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
This is a collection of applications which use this technology.

## Warbler
- [Warbler](https://warblr.net/).
