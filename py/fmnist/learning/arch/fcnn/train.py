import argparse
import logging
import os.path
import pathlib
from typing import Dict, Any, Tuple

import tensorflow as tf

from fmnist import constants
from fmnist.data import metadata
from fmnist.learning import task
from fmnist.learning.arch.fcnn import network

logger = logging.getLogger('tensorflow')


def parse_args() -> argparse.Namespace:
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST FCNN Deep Neural Network')
    arg_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    arg_parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    arg_parser.add_argument('--num-layers', type=int, default=4,
                            help='Use this to create a deep model, so you can see trade-offs in compute vs IO')
    arg_parser.add_argument('--layer-size', type=int, default=512, help='Number of neurons per layer')
    arg_parser.add_argument('--activation', type=str, nargs='?', default='relu')
    arg_parser.add_argument('--num-epochs', type=int, default=1, help='Num training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    arg_parser.add_argument('--optimizer', type=str, default='adamax', choices=('adam', 'adamax', 'nadam', 'rms-prop'))
    arg_parser.add_argument('--buffer-size', type=int, default=1024, help='Capacity for the reading queue')
    arg_parser.add_argument('--num-threads', type=int, default=2, help='Number of threads for processing data')
    arg_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    arg_parser.add_argument('--job-dir', required=True, help='Path to job dir')
    arg_parser.add_argument('--model-dir', required=True, help='Path to model dir')
    arg_parser.add_argument('--train-data', required=True, help='Path to input data path')
    arg_parser.set_defaults(shuffle=True)

    args = arg_parser.parse_args()
    logger.info('Running with args:')
    for arg in vars(args):
        logger.info('\t%s: %s', arg, getattr(args, arg))

    return args


def create_optimizer(choice: str, **params) -> tf.keras.optimizers.Optimizer:
    if choice == 'adam':
        return tf.keras.optimizers.Adam(**params)
    elif choice == 'adamax':
        return tf.keras.optimizers.Adamax(**params)
    elif choice == 'nadam':
        return tf.keras.optimizers.Nadam(**params)
    elif choice == 'rms-prop':
        return tf.keras.optimizers.RMSprop(**params)
    else:
        raise RuntimeError('Unsupported choice of optimizer: %s' % choice)


def train(base_data_dir: str, num_threads: int, buffer_size: int, batch_size: int, num_epochs: int, shuffle: bool,
          job_dir: str, model_dir: str,
          learning_rate: float, dropout_rate: float, activation: str,
          num_layers: int, layer_size: int,
          optimizer_name: str) -> Tuple[Dict[str, Any], pathlib.Path]:
    trn_paths = task.resolve_data_path(base_data_dir, 'train')
    val_paths = task.resolve_data_path(base_data_dir, 'val')

    logger.info('Loading data from %s', base_data_dir)

    trn_dataset = task.build_features(trn_paths,
                                      num_threads=num_threads,
                                      buffer_size=buffer_size,
                                      batch_size=batch_size,
                                      num_epochs=1,
                                      shuffle=shuffle)
    val_dataset = task.build_features(val_paths,
                                      num_threads=num_threads,
                                      buffer_size=buffer_size,
                                      batch_size=batch_size,
                                      num_epochs=1,
                                      shuffle=shuffle)

    # Build the Estimator
    logger.info('Creating model spec')

    optimizer = create_optimizer(optimizer_name, learning_rate=learning_rate)

    m = network.FCNN(dropout_rate=dropout_rate,
                     num_classes=constants.FMNIST_NUM_CLASSES,
                     activation=activation,
                     num_layers=num_layers,
                     optimizer=optimizer,
                     layer_size=layer_size,
                     label_index=metadata.LABEL_INDEX,
                     label_weights=metadata.LABEL_WEIGHTS,
                     num_threads=num_threads)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=job_dir, histogram_freq=1, update_freq='epoch')
    callbacks = [tensorboard_callback]

    logger.info('Starting training')

    # verbose=2 logs per epoch
    m.fit(trn_dataset, epochs=num_epochs, callbacks=callbacks, verbose=constants.TF_LOG_PER_EPOCH)
    results = m.evaluate(val_dataset, callbacks=callbacks, verbose=constants.TF_LOG_PER_BATCH)
    loss, metrics_values = results[0], results[1:]

    metrics = {metric.name: metrics_values[i] for i, metric in enumerate(m.metrics)}
    metrics['loss'] = loss
    for name, value in metrics.items():
        logger.info('%s -> %s', name, value)

    task.export_model(m, model_dir)

    return metrics, pathlib.Path(model_dir)


def main():
    """
    Runs training and evaluating of ƒmnist on a neural network
    :return:
    """
    args = parse_args()
    base_data_dir = os.path.join(args.train_data, constants.DataPaths.INTERIM)
    metrics, export_path = train(base_data_dir, num_threads=args.num_threads, buffer_size=args.buffer_size,
                                 batch_size=args.batch_size, num_epochs=args.num_epochs, shuffle=args.shuffle,
                                 job_dir=args.job_dir, model_dir=args.model_dir,
                                 learning_rate=args.lr, dropout_rate=args.dropout_rate, activation=args.activation,
                                 num_layers=args.num_layers, layer_size=args.layer_size,
                                 optimizer_name=args.optimizer)


if __name__ == '__main__':
    main()
