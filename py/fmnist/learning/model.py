from typing import Dict, Any, List, Optional

import tensorflow as tf

from fmnist import xmath, constants


class FCNN(object):
    """
    Fully Connected Neural Network
    """

    @classmethod
    def feature_columns_spec(cls) -> List[Any]:
        embedding_size = xmath.SeqOp.multiply(constants.FMNIST_EMBEDDING_DIMENSIONS)
        return [tf.feature_column.numeric_column('image_embedding',
                                                 shape=(embedding_size,))]

    @classmethod
    def feature_spec(cls) -> Dict[str, Any]:
        return {
            'image_embedding': tf.io.FixedLenFeature([xmath.SeqOp.multiply(constants.FMNIST_EMBEDDING_DIMENSIONS)],
                                                     tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

    @classmethod
    def create_model(cls, job_dir: str, learning_rate: float, dropout_rate: float, num_classes: int,
                     activation: Optional[str], num_layers: int, layer_size: int, label_index: Dict[str, int],
                     label_weights: Dict[str, float]) -> tf.keras.models.Model:
        """
        Creates a model function
        :return: model_fn of type (features_dict, labels, mode) -> :class:`tf.estimator.EstimatorSpec`
        """
        feature_layer = tf.keras.layers.DenseFeatures(cls.feature_columns_spec(), name='input')

        layers = [feature_layer]

        for _ in range(num_layers):
            layers.append(tf.keras.layers.Dense(layer_size, activation=activation))
            layers.append(tf.keras.layers.Dropout(dropout_rate))

        final_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax')
        layers.append(final_layer)

        tf.summary.histogram('layer-softmax', final_layer.variables)

        model = tf.keras.Sequential(layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model
