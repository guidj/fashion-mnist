from typing import Optional, Dict, List, Any

import tensorflow as tf

from fmnist import xmath, constants
from fmnist.learning.arch import base


class FCNN(base.BaseModel):
    """
    Fully connected neural network.
    """
    def __init__(self, dropout_rate: float, num_classes: int, activation: Optional[str],
                 num_layers: int, layer_size: int, optimizer: tf.keras.optimizers.Optimizer,
                 label_index: Dict[str, int], label_weights: Dict[str, float],
                 num_threads: int):
        """
        :param dropout_rate: Dropout rate for each layer
        :param num_classes: Number of classes in classifier
        :param activation: Choice as a string (e.g. elu)
        :param num_layers: Number of layers in the network
        :param layer_size: Number of neurons per fully connected layer
        :param optimizer: Choice as a string (e.g. adamax)
        :param label_index: Map of string -> index (e.g. house -> 1)
        :param label_weights: Map of string -> weight (e.g. dog -> 0.7)
        :param num_threads: For internal data processing steps
        """
        self.num_threads = num_threads
        self.label_index = label_index
        self.label_weights = label_weights

        feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns_spec(), name='input')

        layers = [feature_layer]

        for _ in range(num_layers):
            layers.append(tf.keras.layers.Dense(layer_size, activation=activation))
            layers.append(tf.keras.layers.Dropout(dropout_rate))

        final_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax')
        layers.append(final_layer)

        tf.summary.histogram('layer-softmax', final_layer.variables)

        self._m = tf.keras.Sequential(layers)
        self._m.compile(optimizer=optimizer,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def preproc(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        No pre-processing required.
        """
        return ds

    def feature_columns_spec(self) -> List[Any]:
        image_size = xmath.SeqOp.multiply(constants.FMNIST_L_DIMENSIONS)
        return [tf.feature_column.numeric_column('image',
                                                 shape=(image_size,), )]

    def fit(self, train_ds: tf.data.Dataset, epochs: int, callbacks: List[tf.keras.callbacks.Callback],
            verbose: int, val_ds: tf.data.Dataset = None, val_epoch_freq: int = 1) -> tf.keras.callbacks.History:
        return self._m.fit(self.preproc(train_ds), epochs=epochs, callbacks=callbacks, verbose=verbose,
                           validation_data=val_ds, validation_freq=val_epoch_freq)

    def evaluate(self, ds: tf.data.Dataset, callbacks, verbose: int) -> List[float]:
        return self._m.evaluate(ds, callbacks=callbacks)

    def export(self, path: str):
        self._m.save(path)

    @property
    def metrics(self) -> List[tf.metrics.Metric]:
        return self._m.metrics
