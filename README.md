fashion-mnist
==============================

Prediction Fashion-MNIST dataset, using embeddings from VGG19.

Note: scripts are designed to run on GCP. Most have a local version though, e.g. `./sbin/task-local-build-features.sh`.


## GCP Config

Copy the file `./sbin/.env` to `./sbin/env` and define the params.

## Data

Downloads data
```sh
$ bash ./sbin/data-download-fashion-mnist.sh
```

Builds features
```sh
$ bash ./sbin/build-features.sh
```

## Training

Train model using embeddings
```sh
$ bash ./sbin/train.sh
```



