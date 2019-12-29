import abc
from typing import Any, List, Tuple

import tensorflow as tf


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def preproc(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        raise NotImplementedError

    @abc.abstractmethod
    def feature_columns_spec(self) -> List[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, ds: tf.data.Dataset, epochs: int, callbacks: List[tf.keras.callbacks.Callback],
            verbose: int) -> tf.keras.callbacks.History:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, ds: tf.data.Dataset, callbacks: List[tf.keras.callbacks.Callback], verbose: int) -> List[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def metrics(self) -> List[tf.metrics.Metric]:
        raise NotImplementedError


class FlatTo3DImageLayer(tf.keras.layers.Layer):
    def __init__(self, image_2d_dimensions: Tuple[int, int], name: str = 'FlatTo3DImageLayer', **kwargs):
        """
        Converts a flat vector to a 3D image. This is designed to convert monochrome images into their
        3-Channel version.
        E.g. Given input of dimensions (batch, 400), containing a batch of images sized 20x20 flattened out,
        this layer can convert that input into a tensor of dimensions (batch, 20, 20, 3).
        To get all three layers, the base layer is replicated.
        An example use case is for use with pre-trained networks that expect a 3 channel input.
        :param image_2d_dimensions: 2D size of image, e.g. 20x20.
        """
        self.image_2d_dimensions = image_2d_dimensions
        super(FlatTo3DImageLayer, self).__init__(trainable=False, name=name, **kwargs)

    def build(self, input_shape):
        super(FlatTo3DImageLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        First, we stack the last dimension
        e.g. data = [[0, 64, 128, 255], [16, 32, 96, 150]]
        [data]*3 =>
             [[[0, 64, 128, 255], [16, 32, 96, 150]],
             [[0, 64, 128, 255], [16, 32, 96, 150]],
             [[0, 64, 128, 255], [16, 32, 96, 150]]]
        tf.stack([data]*3, axis=-1) =>
            <tf.Tensor: id=122, shape=(2, 4, 3), dtype=int32, numpy=
                array([[[  0,   0,   0],
                        [ 64,  64,  64],
                        [128, 128, 128],
                        [255, 255, 255]],
                       [[ 16,  16,  16],
                        [ 32,  32,  32],
                        [ 96,  96,  96],
                        [150, 150, 150]]], dtype=int32)>

        tf.reshape(tf.stack([data]*3, axis=-1), shape=(-1,) + (2, 2) + (3,)) =>
            <tf.Tensor: id=129, shape=(2, 2, 2, 3), dtype=int32, numpy=
            array([[[[  0,   0,   0],
                     [ 64,  64,  64]],
                    [[128, 128, 128],
                     [255, 255, 255]]],
                   [[[ 16,  16,  16],
                     [ 32,  32,  32]],
                    [[ 96,  96,  96],
                     [150, 150, 150]]]], dtype=int32)>
        """
        x_stacked = tf.stack([x] * 3, axis=-1)
        return tf.reshape(x_stacked, shape=(-1,) + self.image_2d_dimensions + (3,))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.image_2d_dimensions + (3,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'image_2d_dimensions': self.image_2d_dimensions
        })
        return config
