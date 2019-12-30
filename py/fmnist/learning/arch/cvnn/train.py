import argparse
import logging
import os.path
import pathlib
from typing import Dict, Any, Tuple

import tensorflow as tf

from fmnist import constants
from fmnist.data import metadata
from fmnist.learning import task
from fmnist.learning.arch.cvnn import network

logger = logging.getLogger('tensorflow')


def parse_args() -> argparse.Namespace:
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST ConvStrix Deep Neural Network')
    arg_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    arg_parser.add_argument('--num-blocks', type=int, default=3,
                            help='Number of convolutional blocks')
    arg_parser.add_argument('--block-size', type=int, default=3,
                            help='Number of convolutional layers per block')
    arg_parser.add_argument('--fcl-num-layers', type=int, default=4,
                            help='Number of layers in the fully connected block')
    arg_parser.add_argument('--fcl-layer-size', type=int, default=512,
                            help='Number of neurons per layer in the fully connected block')
    arg_parser.add_argument('--fcl-dropout-rate', type=float, default=0.3,
                            help='Dropout rate for the fully connected block')
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
          learning_rate: float, activation: str,
          num_blocks: int, block_size: int,
          fcl_num_layers: int, fcl_layer_size: int, fcl_dropout_rate: float,
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

    m = network.ConvStrix(
        num_classes=constants.FMNIST_NUM_CLASSES,
        activation=activation,
        num_blocks=num_blocks,
        block_size=block_size,
        fcl_num_layers=fcl_num_layers,
        fcl_layer_size=fcl_layer_size,
        fcl_dropout_rate=fcl_dropout_rate,
        optimizer=optimizer,
        label_index=metadata.LABEL_INDEX,
        label_weights=metadata.LABEL_WEIGHTS,
        num_threads=num_threads)

    callbacks = task.create_default_callbacks(job_dir)

    logger.info('Starting training')

    m.fit(trn_dataset, epochs=num_epochs, callbacks=callbacks, verbose=constants.TF_LOG_PER_EPOCH,
          val_ds=val_dataset)
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
    Runs training and evaluating of ƒmnist on a convolutional neural network
    :return:
    """
    args = parse_args()
    base_data_dir = os.path.join(args.train_data, constants.DataPaths.INTERIM)
    metrics, export_path = train(base_data_dir, num_threads=args.num_threads, buffer_size=args.buffer_size,
                                 batch_size=args.batch_size, num_epochs=args.num_epochs, shuffle=args.shuffle,
                                 job_dir=args.job_dir, model_dir=args.model_dir,
                                 learning_rate=args.lr, activation=args.activation,
                                 num_blocks=args.num_blocks, block_size=args.block_size,
                                 fcl_dropout_rate=args.fcl_dropout_rate,
                                 fcl_num_layers=args.fcl_num_layers, fcl_layer_size=args.fcl_layer_size,
                                 optimizer_name=args.optimizer)


if __name__ == '__main__':
    main()
