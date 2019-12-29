import argparse
import logging
import os.path

import tensorflow as tf
import numpy as np

from fmnist import constants
from fmnist.data import metadata
from fmnist.learning import task

logger = logging.getLogger('tensorflow')


def parse_args():
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST Prediction')
    arg_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    arg_parser.add_argument('--buffer-size', type=int, default=1024, help='Capacity for the reading queue')
    arg_parser.add_argument('--num-threads', type=int, default=1, help='Number of threads for processing data')
    arg_parser.add_argument('--model-dir', required=True, help='Path to job dir')
    arg_parser.add_argument('--train-data', required=True, help='Path to input data path')

    args = arg_parser.parse_args()
    logger.info("Running with args:")
    for arg in vars(args):
        logger.info("\t%s: %s", arg, getattr(args, arg))

    return args


def generate_classification_report(y_true, y_pred, target_names):
    from sklearn.metrics import classification_report
    logger.info(classification_report(y_true, y_pred=y_pred, target_names=target_names))


def main():
    """
    Runs training and evaluating of ƒmnist on a neural network
    :return:
    """
    args = parse_args()

    base_data_dir = os.path.join(args.train_data, constants.DataPaths.INTERIM)
    tst_path = task.resolve_data_path(base_data_dir, 'test')

    logger.info('Loading data from %s', base_data_dir)
    tst_dataset = task.build_features(tst_path,
                                      num_threads=args.num_threads,
                                      buffer_size=args.buffer_size,
                                      batch_size=args.batch_size,
                                      num_epochs=1)

    model = task.load_model(args.model_dir)
    logger.info(model.summary())

    labels = []
    predictions = []
    for features, label in tst_dataset:
        prediction = model.predict(features)
        labels.extend(label)
        predictions.extend(np.argmax(prediction, axis=1))

    generate_classification_report(labels, predictions, metadata.LABEL_NAMES)


if __name__ == '__main__':
    main()
