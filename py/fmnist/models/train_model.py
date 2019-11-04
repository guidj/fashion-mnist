from __future__ import print_function
import logging
import argparse
import os.path

import tensorflow as tf
import numpy as np
from typing import Dict, Any

from fmnist import constants
from fmnist import xmath

logger = logging.getLogger('tensorflow')

PROFILE_FILENAME = 'profile.pb'


def neural_net(x_dict: Dict[str, np.ndarray], num_classes: int, activation_fn: Any, num_layers: int,
               dropout_rate: float, training: bool):
    with tf.name_scope('input-vgg19-embedding'):
        prev_layer = x_dict['image_embedding']
        tf.summary.histogram('input', prev_layer)

    with tf.name_scope('fcnn'):
        for i in range(num_layers):
            prev_layer = tf.layers.dense(prev_layer, 256, activation=activation_fn)
            prev_layer = tf.layers.dropout(prev_layer, rate=dropout_rate, training=training)
            tf.summary.histogram('layer-{}'.format(i), prev_layer)

        out_layer = tf.layers.dense(prev_layer, num_classes)
        tf.summary.histogram('layer-final', out_layer)
    return out_layer


# Define the model function (following TF Estimator Template)
def create_model_fn(model_dir: str, learning_rate: float, dropout_rate: float, num_classes: int, activation: bool,
                    num_layers: int):
    """
    Creates a model function
    :return: model_fn of type (features_dict, labels, mode) -> :class:`tf.estimator.EstimatorSpec`
    """

    def fn(features, labels, mode):
        # Build the neural network
        if activation:
            activation_fn = tf.nn.relu
        else:
            activation_fn = None

        logits = neural_net(features, num_classes, activation_fn, num_layers, dropout_rate,
                            training=mode == tf.estimator.ModeKeys.TRAIN)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probas = tf.nn.softmax(logits)

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(pred_probas)
        }

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes, export_outputs=export_outputs)

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc)
        tf.summary.histogram('pred-softmax', pred_probas)

        summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                 output_dir=model_dir,
                                                 summary_op=tf.summary.merge_all())

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={'accuracy': (acc, acc_op)},
            export_outputs=export_outputs,
            evaluation_hooks=[summary_hook]
        )

        return estimator_specs

    return fn


def feature_spec():
    return {
        'image_embedding': tf.FixedLenFeature([xmath.SeqOp.multiply(constants.FMNIST_EMBEDDING_DIMENSIONS)], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64)
    }


def build_features(x_path: str, y_path: str, batch_size: int, num_threads: int, queue_capacity: int,
                   num_epochs: int = None, shuffle: bool = False):
    """
    Note: data coming from max-pool layer is already normalized across features
    """

    def load(path):
        with tf.io.gfile.GFile(path, 'rb') as fp:
            with np.load(fp) as data:
                return data[data.files[0]]

    # def transform_fn(x, y):
    # return {'image_embedding': x}, y

    x = load(x_path)
    y = load(y_path)
    x = np.reshape(x, (x.shape[0], xmath.SeqOp.multiply(constants.FMNIST_EMBEDDING_DIMENSIONS)))
    # normalize values values -- hack around not having the actual max;
    # values should fall between [0...U]
    x = x/80.
    y = np.argmax(y, axis=1)

    return tf.estimator.inputs.numpy_input_fn(
        x={'image_embedding': x}, y=y,
        batch_size=batch_size,
        num_threads=num_threads,
        queue_capacity=queue_capacity,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


def resolve_data_path(basedir: str, phase: str):
    def resolve(asset):
        return os.path.join(basedir, '{}_{}.npz'.format(phase, asset))

    return resolve('features'), resolve('label')


def parse_args():
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST (VGG19 Embeddings) Deep Neural Network')
    arg_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    arg_parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    arg_parser.add_argument('--num-layers', type=int, default=4,
                            help='Use this to create a deep model, so you can see trade-offs in compute vs IO')
    arg_parser.add_argument('--num-epochs', type=int, default=1, help='Num training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    arg_parser.add_argument('--queue-capacity', type=int, default=1024, help='Capacity for the reading queue')
    arg_parser.add_argument('--num-threads', type=int, default=1, help='Number of threads for processing data')
    arg_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    arg_parser.add_argument('--no-activation', dest='activation', action='store_false')
    arg_parser.add_argument('--model-dir', required=True, help='Path to model dir')
    arg_parser.add_argument('--train-data', required=True, help='Path to input data path')
    arg_parser.add_argument('--debug', dest='debug', action='store_true', help='Run with tfdbg')
    arg_parser.set_defaults(activation=True, shuffle=True, debug=False)

    args = arg_parser.parse_args()
    logger.info("Running with args:")
    for arg in vars(args):
        logger.info("\t%s: %s", arg, getattr(args, arg))

    return args


def main():
    """
    Runs training and evaluating of ƒmnist on a neural network
    :return:
    """
    args = parse_args()
    base_data_dir = os.path.join(args.train_data, constants.DataPaths.INTERIM)
    (trn_x_path, trn_y_path) = resolve_data_path(base_data_dir, 'train')
    (tst_x_path, tst_y_path) = resolve_data_path(base_data_dir, 'test')

    logger.info('Loading data from %s', base_data_dir)

    trn_input_fn = build_features(trn_x_path, trn_y_path,
                                  num_threads=args.num_threads,
                                  queue_capacity=args.queue_capacity,
                                  batch_size=args.batch_size,
                                  num_epochs=args.num_epochs,
                                  shuffle=args.shuffle)
    tst_input_fn = build_features(tst_x_path, tst_y_path,
                                  num_threads=args.num_threads,
                                  queue_capacity=args.queue_capacity,
                                  batch_size=args.batch_size,
                                  num_epochs=args.num_epochs,
                                  shuffle=args.shuffle)

    # with tf.Session() as sess:
    logger.info('Creating model spec')

    # Build the Estimator
    model_fn = create_model_fn(model_dir=args.model_dir,
                               learning_rate=args.lr,
                               dropout_rate=args.dropout_rate,
                               num_classes=constants.FMNIST_NUM_CLASSES,
                               activation=args.activation,
                               num_layers=args.num_layers)

    model = tf.estimator.Estimator(model_fn, args.model_dir)

    # Train the Model
    hooks = []
    if args.debug:
        from tensorflow.python import debug as tf_debug
        hooks.append(tf_debug.LocalCLIDebugHook())

    logger.info('Starting training')

    model.train(trn_input_fn, hooks=hooks)

    # Run evaluation
    e = model.evaluate(tst_input_fn)

    logger.info('Testing Accuracy: %s', e['accuracy'])

    model.export_saved_model(args.model_dir, serving_input_receiver_fn=create_serving_input_receiver_fn())


def create_serving_input_receiver_fn():
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec())


if __name__ == '__main__':
    main()
