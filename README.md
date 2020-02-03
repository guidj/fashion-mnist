fashion-mnist
==============================

Prediction [Fashion-MNIST dataset](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/), using a few different models.

Note: some parts of this task can run on GCP's AI-platform, but as of this writing (January 2020), tensorflow 2.0 isn't technically
support. CPU based computations, however, run just fine. But GPU is rather difficult, since CUDA versions between
tensorflow 2.0 and AI-platform are currently misaligned -- as I said, tensorflow 2.0 isn't technically supported.

So either deploy a GCE instance with the needed tooling or use an alternative computing environment (e.g. kubeflow).


## GCP Config

Copy the file `./sbin/.env` to `./sbin/env` and define the params.

## Data

Downloads data
```sh
$ bash ./sbin/data-download-fashion-mnist.sh
```

Builds features
```sh
$ bash ./sbin/build-local-features.sh
```

## Training

Each model has its sub folder of scripts to run training and hyper-parameter tuning.

For training it's:
```sh
$ bash ./sbin/{model}/task-local-train.sh
```
And for hyper-parameter tuning it's:
```sh
$ bash ./sbin/{model}/task-local-hps.sh
```

## Models

All models share a common data pre-processing steps:

  - Images are resized from 28x28 to 128x128. This is to enable deeper convolutional which will reduce the size of the
  image from block to block.
  - Pixel values are normalized to [0, 1] - from [0, 255].
  - Labels are exported with the images as numpy arrays.

The following models are implemented:

### FCNN
Standard fully connected neural network. Uses images as is, i.e. flattened arrays.


### CVNN

Convolutional network, inspired by VGG. It uses blocks with sequential convolutional steps, followed by
normalization, and pooling.

### VGG19-FCNN [WIP]

A model that uses embeddings from VGG19 and feeds them to a fully connected block. Basically, a transfer learning
model.
