import os.path

import tensorflow as tf


def prepare_path(path: str, clean: bool = False) -> None:
    if clean and tf.io.gfile.exists(path):
        tf.io.gfile.rmtree(path)
    elif not tf.io.gfile.exists(os.path.dirname(path)):
        tf.io.gfile.makedirs(os.path.dirname(path))
