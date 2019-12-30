import glob
import logging
import os.path
import time
from typing import Optional, List

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras import models

from fmnist import xmath, constants, xpath
from fmnist.learning.arch import base

logger = logging.getLogger('tensorflow')


class EpochDuration(tf.keras.callbacks.Callback):
    """
    Logs epoch duration with a given identifier
    """
    def __init__(self, identifier: str):
        self.__start = None
        self.identifier = identifier
        super(EpochDuration, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.__start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logger.info('Epoch duration for %s -> %.2fs', self.identifier, time.time() - self.__start)


def create_default_callbacks(job_dir: str) -> List[tf.keras.callbacks.Callback]:
    """
    Creates a set of default callbacks
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=job_dir, histogram_freq=1, update_freq='epoch')
    epoch_duration_callback = EpochDuration(identifier=job_dir)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=3)
    return [tensorboard_callback, epoch_duration_callback, early_stopping_callback]


def build_features(paths: List[str], batch_size: int, num_threads: int, buffer_size: int,
                   num_epochs: int = None, shuffle: bool = False):
    """
    Note: data coming from max-pool layer is already normalized across features
    """

    def load():
        for path in paths:
            with tf.io.gfile.GFile(path, 'rb') as fp:
                with np.load(fp) as data:
                    xs, ys = data[data.files[0]], data[data.files[1]]
                    for i in range(xs.shape[0]):
                        yield xs[i], ys[i]

    def process_fn(features, label):
        return {'image': features}, label

    dataset = tf.data.Dataset.from_generator(load,
                                             output_types=(tf.float32, tf.int32),
                                             output_shapes=([xmath.SeqOp.multiply(constants.FMNIST_L_DIMENSIONS)], []))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size)
    dataset = dataset.map(process_fn, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size if shuffle is True else batch_size)
    return dataset


def resolve_data_path(basedir: str, phase: str) -> List[str]:
    return glob.glob(os.path.join(basedir, phase, 'part-*'))


def export_model(model: base.BaseModel, export_dir: str) -> None:
    path = os.path.join(export_dir, str(int(time.time())))
    xpath.prepare_path(path, clean=True)
    logger.info('Saving to path %s', path)
    model.export(path)


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
