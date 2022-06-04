# import keras.saving.saved_model.model_serialization
import sklearn.model_selection
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets, optimizers
import numpy as np

# Manually defined models
import BaseCNN
import Ensemble

import custom_callbacks


def load_data():
    # (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(
        x_train, y_train, epochs=10, verbose=1, validation_split=0.2,
        callbacks=custom_callbacks.get_callbacks()
    )

    print("=================================================================")
    _, acc = model.evaluate(x_test, y_test, verbose=1)


def main():
    x_train, y_train, x_test, y_test = load_data()

    optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.8)

    # Change to Ensemble.get_model() for other model
    model = BaseCNN.get_model(
        input_shape=x_train[0][:,:,None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
        n_outputs=y_train[0].shape[0],
        optimizer=optimizer,
        dropout=True
    )
    print(model.summary())

    train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )


def split_data(x, y, n_nets, seed=42):
    np.random.seed(seed=seed)
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    idx_blocks = np.split(idx, n_nets)
    return np.take(x, idx_blocks), np.take(y, idx_blocks)


def ensemble_main():
    n_nets = 5
    seed=42
    x_train, y_train, x_test, y_test = load_data()
    rs = StratifiedShuffleSplit(n_splits=n_nets, test_size=0.2, random_state=seed)
    for train_index, test_index in rs.split(x_train, y_train):

        x_train_block, y_train_block = x_train[train_index], y_train[train_index]
        x_test_block, y_test_block = x_train[test_index], y_train[test_index]
        print(f'TRAIN: {x_train_block[0]}, TEST: {x_test_block[0]}')
        print(f'{x_train_block[0].shape}, {x_test_block[0].shape}')
if __name__ == "__main__":
    #main()
    ensemble_main()
