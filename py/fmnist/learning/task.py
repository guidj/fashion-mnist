import logging
import os.path
import time
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras import models

from fmnist import xmath, constants, xpath

logger = logging.getLogger('tensorflow')


def build_features(x_path: str, y_path: str, batch_size: int, num_threads: int, buffer_size: int,
                   num_epochs: int = None, shuffle: bool = False):
    """
    Note: data coming from max-pool layer is already normalized across features
    """

    def load(path):
        with tf.io.gfile.GFile(path, 'rb') as fp:
            with np.load(fp) as data:
                return data[data.files[0]]

    def process_fn(features, label):
        return {'image_embedding': features}, label

    x = load(x_path)
    y = load(y_path)
    x = np.reshape(x, (x.shape[0], xmath.SeqOp.multiply(constants.FMNIST_EMBEDDING_DIMENSIONS)))
    # normalize values values -- hack around not having the actual max;
    # values should fall between [0...U]
    x = x / 80.
    y = np.argmax(y, axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size)
    dataset = dataset.map(process_fn, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size if shuffle is True else batch_size)
    return dataset


def resolve_data_path(basedir: str, phase: str):
    def resolve(asset):
        return os.path.join(basedir, '{}_{}.npz'.format(phase, asset))

    return resolve('features'), resolve('label')


def export_model(model: models.Model, export_dir: str) -> None:
    path = os.path.join(export_dir, str(int(time.time())))
    xpath.prepare_path(path, clean=True)
    logger.info('Saving to path %s', path)
    model.save(path)


def load_model(path: str) -> Optional[models.Model]:
    if tf.io.gfile.exists(os.path.join(path, 'saved_model.pb')):
        return tf.keras.models.load_model(path)
    else:
        files = tf.io.gfile.listdir(path)
        paths = [subpath for subpath in [os.path.join(path, file) for file in files] if tf.io.gfile.isdir(subpath)]
        paths = sorted(paths, reverse=True)

        try:
            return tf.keras.models.load_model(next(iter(paths)))
        except StopIteration:
            return None
