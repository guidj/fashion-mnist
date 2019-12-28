import abc
from typing import Dict, Any, List, Optional

import tensorflow as tf

from fmnist import xmath, constants


class LTModel(abc.ABC):
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


class FCNN(LTModel):
    def __init__(self, dropout_rate: float, num_classes: int, activation: Optional[str],
                 num_layers: int, layer_size: int, optimizer: tf.keras.optimizers.Optimizer,
                 label_index: Dict[str, int], label_weights: Dict[str, float],
                 num_threads: int):
        """
        Creates a model function
        :return: model_fn of type (features_dict, labels, mode) -> :class:`tf.estimator.EstimatorSpec`
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
                        loss='categorical_crossentropy',
                        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    def preproc(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds

    def feature_columns_spec(self) -> List[Any]:
        image_size = xmath.SeqOp.multiply(constants.FMNIST_UP_DIMENSIONS)
        return [tf.feature_column.numeric_column('image',
                                                 shape=(image_size,), )]

    def fit(self, ds: tf.data.Dataset, epochs: int, callbacks: List[tf.keras.callbacks.Callback],
            verbose: int) -> tf.keras.callbacks.History:
        return self._m.fit(self.preproc(ds), epochs=epochs, callbacks=callbacks, verbose=verbose)

    def evaluate(self, ds: tf.data.Dataset, callbacks, verbose: int) -> List[float]:
        return self._m.evaluate(ds, callbacks=callbacks)

    def export(self, path: str):
        self._m.save(path)

    @property
    def metrics(self) -> List[tf.metrics.Metric]:
        return self._m.metrics
