import multiprocessing as mp
import os
import os.path
import argparse
from typing import Tuple, Callable

import cv2
import keras
import keras.utils
import numpy as np
import pandas as pd
from keras.applications import vgg19
from sklearn import model_selection
import tensorflow as tf

from fmnist import logger
from fmnist.constants import DataPaths

MP_THREADS = mp.cpu_count()

IMAGE_SIZE = 128
NUM_CLASSES = 10

DataTuple = Tuple[np.ndarray, np.ndarray]


def reshape(img_array: np.ndarray) -> np.ndarray:
    return img_array.reshape(-1, 28)


def create_path_fn(base: str) -> Callable[[str, str], str]:
    def fn(path: str, filename: str) -> str:
        return os.path.join(base, path, filename)

    return fn


def load_data_fn(train_path: str, test_path: str) -> Tuple[DataTuple, DataTuple, DataTuple]:
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)

    # X forms the training images, and y forms the training labels
    X = np.array(data_train.iloc[:, 1:])
    # TODO: save without OHE -- doesn't take much space but isn't useful either
    y = keras.utils.to_categorical(np.array(data_train.iloc[:, 0]))

    # X_test forms the test images, and y_test forms the test labels
    X_test = np.array(data_test.iloc[:, 1:])
    y_test = keras.utils.to_categorical(np.array(data_test.iloc[:, 0]))

    # Convert the training and test images into 3 channels
    logger.info('Convert images to 3 channels')
    X = np.dstack([X] * 3)
    X_test = np.dstack([X_test] * 3)
    logger.info('%s, %s', X.shape, X_test.shape)

    logger.info('Reshape images')
    X = X.reshape(-1, 28, 28, 3)
    X_test = X_test.reshape(-1, 28, 28, 3)
    logger.info('%s, %s', X.shape, X_test.shape)

    # We'll resize the images using OpenCV-2. It provides a faithful upsizing of the images.
    logger.info('Resize images')
    X = X.astype('uint8')
    X_test = X_test.astype('uint8')
    X = np.asarray([cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE)) for im in X])
    X_test = np.asarray([cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE)) for im in X_test])
    logger.info('%s, %s', X.shape, X_test.shape)

    # Normalise the data and change data type
    fX = X.astype('float32')
    fX /= 255

    fX_test = X_test.astype('float32')
    fX_test /= 255

    fX_train, fX_val, y_train, y_val = model_selection.train_test_split(fX, y, test_size=0.2, shuffle=True,
                                                                        random_state=13)

    return (fX_train, y_train), (fX_val, y_val), (fX_test, y_test)


def load_model_fn(image_size: int, num_classes: int):
    from keras.applications import VGG19
    # Create the base model of VGG19
    return VGG19(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3), classes=num_classes)


def embeddings_fn(model: keras.Model, batch_size: int, *inputs):
    # fn_learned_layer = K.function([model_vgg19.layers[0].input], [model_vgg19.layers[1].output])

    predictions = []
    for input_ in inputs:
        input_preproc = vgg19.preprocess_input(input_)
        embeddings = model.predict(np.array(input_preproc), batch_size=batch_size, verbose=1)
        predictions.append(embeddings)

    return predictions


def export_fn(path: str, array: np.ndarray) -> None:
    import tempfile
    import uuid
    temporary_path = os.path.join(tempfile.gettempdir(), '{}.npz'.format(uuid.uuid4()))

    if not tf.io.gfile.exists(os.path.dirname(path)):
        tf.io.gfile.makedirs(os.path.dirname(path))
    with open(temporary_path, 'wb') as fp:
        np.savez(fp, array)

    logger.info('Copying %s to %s', temporary_path, path)
    tf.io.gfile.copy(src=temporary_path, dst=path, overwrite=True)


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

    (fX_train, y_train), (fX_val, y_val), (fX_test, y_test) = \
        load_data_fn(train_path=fpath(DataPaths.FMNIST, 'fashion-mnist_train.csv'),
                     test_path=fpath(DataPaths.FMNIST, 'fashion-mnist_test.csv'))

    # Check the data size whether it is as per tensorflow and VGG19 requirement
    logger.info('X: (%s), X_val: (%s), y: (%s), y_val: (%s)', fX_train.shape, fX_val.shape, y_train.shape, y_val.shape)

    logger.info('Loading VGG19')
    model_vgg19 = load_model_fn(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

    emb_train, emb_val, emd_test = embeddings_fn(model_vgg19, args.batch_size, fX_train, fX_val, fX_test)

    # Saving the features so that they can be used for future
    for prefix, embedding, y in zip(['train', 'val', 'test'], [emb_train, emb_val, emd_test], [y_train, y_val, y_test]):
        export_fn(fpath(DataPaths.INTERIM, '{}_features.npz'.format(prefix)), embedding)
        export_fn(fpath(DataPaths.INTERIM, '{}_label.npz'.format(prefix)), y)


if __name__ == '__main__':
    main()
