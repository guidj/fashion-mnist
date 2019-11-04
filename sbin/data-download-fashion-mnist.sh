donwload_and_unpack() {
    DIR=$(dirname $0)
    PARENT_DIR=$DIR/..

    kaggle datasets download -d zalando-research/fashionmnist -p data/external
    unzip $PARENT_DIR/data/external/fashionmnist.zip -d $PARENT_DIR/data/external/fashion-mnist
}

donwload_and_unpack