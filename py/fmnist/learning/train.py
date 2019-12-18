import argparse
import logging
import os.path
import pathlib
from typing import Dict, Any, Tuple

from fmnist import constants
from fmnist.data import metadata
from fmnist.learning import model
from fmnist.learning import task

logger = logging.getLogger('tensorflow')


def parse_args() -> argparse.Namespace:
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST (VGG19 Embeddings) Deep Neural Network')
    arg_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    arg_parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    arg_parser.add_argument('--num-layers', type=int, default=8,
                            help='Use this to create a deep model, so you can see trade-offs in compute vs IO')
    arg_parser.add_argument('--layer-size', type=int, default=512, help='Number of neurons per layer')
    arg_parser.add_argument('--activation', type=str, nargs='?', dest='activation', default='relu')
    arg_parser.add_argument('--num-epochs', type=int, default=1, help='Num training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    arg_parser.add_argument('--buffer-size', type=int, default=1024, help='Capacity for the reading queue')
    arg_parser.add_argument('--num-threads', type=int, default=1, help='Number of threads for processing data')
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


def train(base_data_dir: str, num_threads: int, buffer_size: int, batch_size: int, num_epochs: int, shuffle: bool,
          job_dir: str, model_dir: str, learning_rate: float, dropout_rate: float, activation: str,
          num_layers: int, layer_size: int) -> Tuple[Dict[str, Any], pathlib.Path]:
    (trn_x_path, trn_y_path) = task.resolve_data_path(base_data_dir, 'train')
    (tst_x_path, tst_y_path) = task.resolve_data_path(base_data_dir, 'test')

    logger.info('Loading data from %s', base_data_dir)

    trn_dataset = task.build_features(trn_x_path, trn_y_path,
                                      num_threads=num_threads,
                                      buffer_size=buffer_size,
                                      batch_size=batch_size,
                                      num_epochs=num_epochs,
                                      shuffle=shuffle)
    tst_dataset = task.build_features(tst_x_path, tst_y_path,
                                      num_threads=num_threads,
                                      buffer_size=buffer_size,
                                      batch_size=batch_size,
                                      num_epochs=1,
                                      shuffle=shuffle)

    # Build the Estimator
    logger.info('Creating model spec')

    m = model.FCNN.create_model(job_dir=job_dir,
                                learning_rate=learning_rate,
                                dropout_rate=dropout_rate,
                                num_classes=constants.FMNIST_NUM_CLASSES,
                                activation=activation,
                                num_layers=num_layers,
                                layer_size=layer_size,
                                label_index=metadata.LABEL_INDEX,
                                label_weights=metadata.LABEL_WEIGHTS)

    logger.info('Starting training')

    # verbose=2 logs per epoch
    m.fit(trn_dataset, verbose=2)
    results = m.evaluate(tst_dataset)
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
                                 job_dir=args.job_dir, model_dir=args.model_dir, learning_rate=args.lr,
                                 dropout_rate=args.dropout_rate, activation=args.activation,
                                 num_layers=args.num_layers, layer_size=args.layer_size)


if __name__ == '__main__':
    main()
