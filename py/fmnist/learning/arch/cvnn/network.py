from typing import Optional, Dict, List, Any

import tensorflow as tf

from fmnist import xmath, constants
from fmnist.learning.arch import base


class ConvStrix(base.BaseModel):
    """
    Convolutional neural architecture inspired by VGG.
    There are blocks of convolution followed by pooling.
    Each block has the same number of filter in each conv layer, and this number for the next block.
    """

    def __init__(self, num_classes: int, activation: Optional[str], num_blocks: int, block_size: int,
                 fcl_num_layers: int, fcl_layer_size: int, fcl_dropout_rate: float,
                 optimizer: tf.keras.optimizers.Optimizer,
                 label_index: Dict[str, int], label_weights: Dict[str, float],
                 num_threads: int):
        """
        :param num_classes: Number of classes in classifier
        :param activation: Choice as a string (e.g. elu)
        :param num_blocks: For the convolutional blocks
        :param block_size: Number of convolutional layers in each block
        :param fcl_num_layers: Number of layers in the fully connected block
        :param fcl_layer_size: Number of neurons per fully connected layer
        :param fcl_dropout_rate: Dropout rate for fully connected layers
        :param optimizer: Choice as a string (e.g. adamax)
        :param label_index: Map of string -> index (e.g. house -> 1)
        :param label_weights: Map of string -> weight (e.g. dog -> 0.7)
        :param num_threads: For internal data processing steps
        """
        self.num_threads = num_threads
        self.label_index = label_index
        self.label_weights = label_weights

        feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns_spec(), name='input')
        i3d_conversion_layer = base.FlatTo3DImageLayer(image_2d_dimensions=constants.FMNIST_L_DIMENSIONS,
                                                       expand_as_rgb=False)

        layers = [feature_layer, i3d_conversion_layer]
        initial_num_filters = 32

        for block_id in range(num_blocks):
            num_filters = initial_num_filters * (block_id + 1)
            for _ in range(block_size):
                layers.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same',
                                                     activation=activation))
            if block_id + 1 < num_blocks:
                layers.append(tf.keras.layers.BatchNormalization())
                layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        layers.append(tf.keras.layers.Flatten())

        for _ in range(fcl_num_layers):
            layers.append(tf.keras.layers.Dense(fcl_layer_size, activation=activation))
            layers.append(tf.keras.layers.Dropout(fcl_dropout_rate))

        final_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax')
        layers.append(final_layer)

        tf.summary.histogram('layer-softmax', final_layer.variables)

        self._m = tf.keras.Sequential(layers)
        self._m.compile(optimizer=optimizer,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def preproc(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        No pre-processing necessary for training data
        """
        return ds

    def feature_columns_spec(self) -> List[Any]:
        image_size = xmath.SeqOp.multiply(constants.FMNIST_L_DIMENSIONS)
        return [tf.feature_column.numeric_column('image',
                                                 shape=(image_size,), )]

    def fit(self, train_ds: tf.data.Dataset, epochs: int, callbacks: List[tf.keras.callbacks.Callback],
            verbose: int,  val_ds: tf.data.Dataset = None, val_epoch_freq: int = 1) -> tf.keras.callbacks.History:
        return self._m.fit(self.preproc(train_ds), epochs=epochs, callbacks=callbacks, verbose=verbose,
                           validation_data=val_ds, validation_freq=val_epoch_freq)

    def evaluate(self, ds: tf.data.Dataset, callbacks, verbose: int) -> List[float]:
        return self._m.evaluate(ds, callbacks=callbacks)

    def export(self, path: str):
        self._m.save(path)

    @property
    def metrics(self) -> List[tf.metrics.Metric]:
        return self._m.metrics
