class DataPaths(object):
    FMNIST = 'external/fashion-mnist'
    EXTERNAL = 'external'
    INTERIM = 'interim'
    PROCESSED = 'processed'


# data
FMNIST_DIMENSIONS = (28, 28)
FMNIST_EMBEDDING_DIMENSIONS = (4, 4, 512)
FMNIST_L_DIMENSIONS = (128, 128)
FMNIST_NUM_CLASSES = 10

# tensorflow
TF_LOG_PER_BATCH = 1
TF_LOG_PER_EPOCH = 2
