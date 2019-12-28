import abc
from typing import Any, List

import tensorflow as tf


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
