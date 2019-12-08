import argparse
import logging
import os.path

from fmnist import constants
from fmnist.models import task
from fmnist.models import model

logger = logging.getLogger('tensorflow')


def parse_args() -> argparse.Namespace:
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST (VGG19 Embeddings) Deep Neural Network')
    arg_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    arg_parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    arg_parser.add_argument('--num-layers', type=int, default=8,
                            help='Use this to create a deep model, so you can see trade-offs in compute vs IO')
    arg_parser.add_argument('--num-epochs', type=int, default=1, help='Num training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    arg_parser.add_argument('--buffer-size', type=int, default=1024, help='Capacity for the reading queue')
    arg_parser.add_argument('--num-threads', type=int, default=1, help='Number of threads for processing data')
    arg_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    arg_parser.add_argument('--no-activation', dest='activation', action='store_false')
    arg_parser.add_argument('--job-dir', required=True, help='Path to model dir')
    arg_parser.add_argument('--model-dir', required=True, help='Path to job dir')
    arg_parser.add_argument('--train-data', required=True, help='Path to input data path')
    arg_parser.add_argument('--debug', dest='debug', action='store_true', help='Run with tfdbg')
    arg_parser.set_defaults(activation=True, shuffle=True, debug=False)

    args = arg_parser.parse_args()
    logger.info('Running with args:')
    for arg in vars(args):
        logger.info('\t%s: %s', arg, getattr(args, arg))

    return args


def main():
    """
    Runs training and evaluating of ƒmnist on a neural network
    :return:
    """
    args = parse_args()
    base_data_dir = os.path.join(args.train_data, constants.DataPaths.INTERIM)
    (trn_x_path, trn_y_path) = task.resolve_data_path(base_data_dir, 'train')
    (tst_x_path, tst_y_path) = task.resolve_data_path(base_data_dir, 'test')

    logger.info('Loading data from %s', base_data_dir)

    trn_dataset = task.build_features(trn_x_path, trn_y_path,
                                      num_threads=args.num_threads,
                                      buffer_size=args.buffer_size,
                                      batch_size=args.batch_size,
                                      num_epochs=args.num_epochs,
                                      shuffle=args.shuffle)
    tst_dataset = task.build_features(tst_x_path, tst_y_path,
                                      num_threads=args.num_threads,
                                      buffer_size=args.buffer_size,
                                      batch_size=args.batch_size,
                                      num_epochs=1,
                                      shuffle=args.shuffle)

    # Build the Estimator
    logger.info('Creating model spec')

    m = model.create_model(job_dir=args.job_dir,
                           learning_rate=args.lr,
                           dropout_rate=args.dropout_rate,
                           num_classes=constants.FMNIST_NUM_CLASSES,
                           activation=args.activation,
                           num_layers=args.num_layers)

    logger.info('Starting training')

    m.fit(trn_dataset)
    results = m.evaluate(tst_dataset)
    for i, metric in enumerate(m.metrics):
        logger.info('%s -> %s', metric.name, results[i])

    task.export_model(m, args.model_dir)


if __name__ == '__main__':
    main()
