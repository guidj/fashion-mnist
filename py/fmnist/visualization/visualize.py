import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector

from fmnist import logger
from fmnist.constants import DataPaths


def parse_args():
    arg_parser = argparse.ArgumentParser('fminst-visualizer', description='Create tensors to viz labels on tensorboard')
    arg_parser.add_argument('--train-data', required=True)
    arg_parser.add_argument('--job-dir', required=False, default=None)

    args = arg_parser.parse_args()

    logger.info('Running with arguments')
    for attr, value in vars(args).items():
        logger.info('%s: %s', attr, value)

    return args


def main():
    args = parse_args()
    base_data_path = os.path.join(args.train_data, DataPaths.FMNIST)
    logs_path = os.path.join(args.train_data, DataPaths.INTERIM, 'fashion-mnist/visualization')
    test_data = np.array(pd.read_csv(os.path.join(base_data_path, 'fashion-mnist_test.csv')), dtype='float32')

    embed_count = 2500

    X_test = test_data[:embed_count, 1:] / 255
    Y_test = test_data[:embed_count, 0]

    # Use this logdir to create a summary writer
    summary_writer = tf.summary.FileWriter(logs_path)
    # Creating the embedding variable with all the images defined above under X_test
    embedding_var = tf.Variable(X_test, name='fmnist_embedding')

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()
    # You can add multiple embeddings. Here I add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(logs_path, 'metadata.tsv')

    # After constructing the sprite, I need to tell the Embedding Projector where to find it
    embedding.sprite.image_path = os.path.join(logs_path, 'sprite.png')
    embedding.sprite.single_image_dim.extend([28, 28])

    # The next line writes a projector_config.pbtxt in the logdir. TensorBoard will read this file during startup.
    projector.visualize_embeddings(summary_writer, config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(logs_path, 'model.ckpt'))

    # create image sprite
    rows = 28
    cols = 28
    label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    sprite_dim = int(np.sqrt(X_test.shape[0]))
    sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim))

    index = 0
    labels = []
    for i in range(sprite_dim):
        for j in range(sprite_dim):
            labels.append(label[int(Y_test[index])])
            sprite_image[i * cols: (i + 1) * cols, j * rows: (j + 1) * rows] = X_test[index].reshape(28, 28) * -1 + 1
            index += 1

    with tf.io.gfile.GFile(embedding.metadata_path, 'w') as meta:
        meta.write('Index\tLabel\n')
        for index, label in enumerate(labels):
            meta.write('{}\t{}\n'.format(index, label))

    with tf.io.gfile.GFile(embedding.sprite.image_path, 'wb') as fp:
        plt.imsave(fp, sprite_image, cmap='gray')


if __name__ == '__main__':
    main()
