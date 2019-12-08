"""
Metrics for multi-class classification
"""
import logging
from typing import List, Optional

import tensorflow as tf

logger = logging.getLogger('tensorflow')


class SingleOutRecall(tf.keras.metrics.Metric):
    """
    Computes recall for a single class using prediction from a multi-class classifier
    """

    def __init__(self, name: str, class_id: int, **kwargs):
        super(SingleOutRecall, self).__init__(name='recall/{}/{}'.format(name, class_id), **kwargs)
        self.class_id = tf.cast(class_id, dtype=tf.int64)
        self.true_positives = tf.keras.metrics.TruePositives()
        self.false_negatives = tf.keras.metrics.FalseNegatives()

    # No support for weights
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Note: Does not support sample_weight at the moment
        """
        y_true = tf.reshape(y_true, shape=(-1,))
        y_true = tf.math.equal(y_true, tf.cast(self.class_id, y_true.dtype))
        y_pred = tf.math.equal(tf.argmax(y_pred, axis=1, output_type=tf.int64), self.class_id)

        self.true_positives.update_state(y_true, y_pred=y_pred)
        self.false_negatives.update_state(y_true, y_pred=y_pred)

    def result(self):
        nom = tf.cast(self.true_positives.result(), tf.float32)
        denom = tf.cast(self.true_positives.result() + self.false_negatives.result(), tf.float32)
        return tf.math.divide_no_nan(nom, denom)

    def reset_states(self):
        self.true_positives.reset_states()
        self.false_negatives.reset_states()


class SingleOutPrecision(tf.keras.metrics.Metric):
    """
    Computes precision for a single class using prediction from a multi-class classifier
    """

    def __init__(self, name: str, class_id: int, **kwargs):
        super(SingleOutPrecision, self).__init__(name='precision/{}/{}'.format(name, class_id), **kwargs)
        self.class_id = tf.cast(class_id, dtype=tf.int64)
        self.true_positives = tf.keras.metrics.TruePositives()
        self.false_positives = tf.keras.metrics.FalsePositives()

    # No support for weights
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Note: Does not support sample_weight at the moment
        """
        y_true = tf.reshape(y_true, shape=(-1,))
        y_true = tf.math.equal(y_true, tf.cast(self.class_id, y_true.dtype))
        y_pred = tf.math.equal(tf.argmax(y_pred, axis=1, output_type=tf.int64), self.class_id)

        self.true_positives.update_state(y_true, y_pred=y_pred)
        self.false_positives.update_state(y_true, y_pred=y_pred)

    def result(self):
        nom = tf.cast(self.true_positives.result(), tf.float32)
        denom = tf.cast(self.true_positives.result() + self.false_positives.result(), tf.float32)
        return tf.math.divide_no_nan(nom, denom)

    def reset_states(self):
        self.true_positives.reset_states()
        self.false_positives.reset_states()


class MultiClassF1(tf.keras.metrics.Metric):
    """
    Computes f1 for a multi-class classifier, using class weights (weighted f1)
    """

    def __init__(self, num_classes: int, class_weights: Optional[List[float]] = None, **kwargs):
        super(MultiClassF1, self).__init__(name='f1', **kwargs)

        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else [1.0 for _ in range(num_classes)]
        self.class_weights = tf.constant(self.class_weights, dtype=tf.float32)
        self.precision = [
            SingleOutPrecision(name='f1/inner/precision/{}'.format(i), class_id=i) for i in range(num_classes)
        ]
        self.recall = [
            SingleOutRecall(name='f1/inner/recall/{}'.format(i), class_id=i) for i in range(num_classes)
        ]

    # No support for weights
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Note: Does not support sample_weight at the moment
        """
        for i in range(self.num_classes):
            self.precision[i].update_state(y_true, y_pred=y_pred)
            self.recall[i].update_state(y_true, y_pred=y_pred)

    def result(self):
        f1 = [0 for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            f1[i] = compute_f1(self.precision[i].result(), self.recall[i].result()) * self.class_weights[i]

        return tf.math.divide_no_nan(tf.reduce_sum(f1), self.num_classes)

    def reset_states(self):
        for i in range(self.num_classes):
            self.precision[i].reset_states()
            self.recall[i].reset_states()


@tf.function
def compute_f1(precision, recall):
    return tf.math.divide_no_nan(precision * recall, precision + recall)
