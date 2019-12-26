import argparse
import multiprocessing as mp
import os
import os.path
from typing import Tuple, Callable, List, Iterator, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection

from fmnist import logger, constants, xmath, xpath
from fmnist.constants import DataPaths

MP_THREADS = mp.cpu_count()

DataTuple = Tuple[np.ndarray, np.ndarray]


def data_frame_split(df: pd.DataFrame, left_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indexes = [v for v in range(len(df))]
    train_idx, test_idx = model_selection.train_test_split(indexes, train_size=left_fraction)
    return df.iloc[train_idx, :], df.iloc[test_idx, :]


def create_path_fn(base: str) -> Callable[[str, str], str]:
    def fn(path: str, filename: str) -> str:
        return os.path.join(base, path, filename)

    return fn


def create_data_generator(x: np.ndarray,
                          y: np.ndarray,
                          processors: List[Callable[[DataTuple], DataTuple]],
                          post_processors: List[Callable[[DataTuple], DataTuple]]) -> Callable[[], Iterable[DataTuple]]:
    assert len(x.shape) == 2, 'x should be a 2-D array'
    n = x.shape[0]

    def post_process(dt: DataTuple, transformers: List[Callable[[DataTuple], DataTuple]]) -> DataTuple:
        if not transformers:
            return dt
        else:
            transformer, transformers = transformers[0], transformers[1:]
            return post_process(transformer(dt), transformers)

    def fn():
        for i in range(n):
            for transformer in processors:
                example_x, example_y = transformer((x[i], y[i]))
                processed_x, processed_y = post_process((example_x, example_y), post_processors)
                yield processed_x, processed_y

    return fn


def create_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    x = np.array(df.iloc[:, 1:])
    y = np.array(df.iloc[:, 0])

    def expand(dt: DataTuple) -> DataTuple:
        x, y = dt
        return np.array([x]), y

    def normalize(dt: DataTuple) -> DataTuple:
        x, y = dt
        return x.astype('float32') / 255., y

    post_processors = [expand]
    generator = create_data_generator(x, y, processors=[normalize], post_processors=post_processors)

    ds = tf.data.Dataset.from_generator(generator,
                                        output_types=(tf.float32, tf.int32),
                                        output_shapes=([1, xmath.SeqOp.multiply(constants.FMNIST_DIMENSIONS)], []))
    return ds


def create_generator(dataset: tf.data.Dataset, batch_size: int) -> Iterator[DataTuple]:
    for x, y in dataset.batch(batch_size):
        yield x, y


def create_partitioning_fn(group_size: int,
                           agg_fn: Callable[[List[DataTuple]], DataTuple],
                           consumer_fn: Callable[[DataTuple, int], None]) -> Callable[[Iterator[DataTuple]], None]:
    """
    :param group_size: group size for values accumulated from the iterator
    :param agg_fn: function to aggregate values in a group
    :param consumer_fn: function to consume result of group aggregation
    """
    assert group_size > 0, 'size should be positive'

    def fn(data: Iterator[DataTuple]):
        group = []
        partition = 0
        for batch in data:
            group.append(batch)
            if len(group) >= group_size:
                # flush
                consumer_fn(agg_fn(group), partition)
                group = []
                partition += 1

        if group:
            # flush
            consumer_fn(agg_fn(group), partition)

    return fn


def create_export_fn(path: str, extension: str) -> Callable[[DataTuple, int], None]:
    def fn(arrays: DataTuple, partition: int) -> None:
        import tempfile
        import uuid
        temporary_path = os.path.join(tempfile.gettempdir(), '{}.{}'.format(uuid.uuid4(), extension))

        file_path = os.path.join(path, 'part-{:03}.{}'.format(partition, extension))
        xpath.prepare_path(file_path)
        with open(temporary_path, 'wb') as fp:
            np.savez(fp, *arrays)

        logger.info('Copying %s to %s', temporary_path, file_path)
        tf.io.gfile.copy(src=temporary_path, dst=file_path, overwrite=True)
    return fn


def agg_fn(dts: List[DataTuple]) -> DataTuple:
    xs, ys = list(zip(*dts))
    xs = np.reshape(np.concatenate(xs, axis=0), newshape=(-1, xmath.SeqOp.multiply(constants.FMNIST_DIMENSIONS)))
    ys = np.concatenate(ys)
    return xs, ys


def parse_args():
    arg_parser = argparse.ArgumentParser('fminst-vgg19-embedding', description='Get VGG19 embeddings for FMNIST')
    arg_parser.add_argument('--train-data', required=True)
    arg_parser.add_argument('--batch-size', required=False, type=int, default=32)
    arg_parser.add_argument('--job-dir', required=False, default=None)

    args = arg_parser.parse_args()

    logger.info('Running with arguments')
    for attr, value in vars(args).items():
        logger.info('%s: %s', attr, value)

    return args


def main():
    args = parse_args()
    fpath = create_path_fn(args.train_data)

    df_prime = pd.read_csv(fpath(DataPaths.FMNIST, 'fashion-mnist_train.csv'))
    df_test = pd.read_csv(fpath(DataPaths.FMNIST, 'fashion-mnist_test.csv'))
    df_train, df_val = data_frame_split(df_prime, left_fraction=0.80)

    for df, split in zip((df_train, df_val, df_test), ('train', 'val', 'test')):
        logger.info('Running partitioning pipeline for %s', split)
        ds = create_dataset(df)
        data_iter = create_generator(ds, batch_size=args.batch_size)

        export_fn = create_export_fn(fpath(DataPaths.INTERIM, split), 'npz')
        partition_export_fn = create_partitioning_fn(group_size=100, agg_fn=agg_fn, consumer_fn=export_fn)
        partition_export_fn(data_iter)


if __name__ == '__main__':
    main()
