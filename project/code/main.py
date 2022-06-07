# import keras.saving.saved_model.model_serialization
import sklearn.model_selection
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets, optimizers
from dataclasses import dataclass
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


def train_model(model, x_train, y_train, x_test, y_test, args):
    history = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_split=args.val_split,
        callbacks=args.callbacks
    )

    print("=================================================================")
    _, acc = model.evaluate(x_test, y_test, verbose=args.verbose)


def main(args):
    x_train, y_train, x_test, y_test = load_data()

    # Change to Ensemble.get_model() for other model
    model = BaseCNN.get_model(
        input_shape=x_train[0][:,:,None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
        n_outputs=y_train[0].shape[0],
        optimizer=args.optimizer,
        dropout=True
    )
    print(model.summary())

    train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        args=args
    )


# def split_data(x, y, n_nets, seed=42):
#     np.random.seed(seed=seed)
#     idx = np.arange(x.shape[0])
#     np.random.shuffle(idx)
#     idx_blocks = np.split(idx, n_nets)
#     return np.take(x, idx_blocks), np.take(y, idx_blocks)


def ensemble_main(args):
    x_train, y_train, x_test, y_test = load_data()
    rs = StratifiedShuffleSplit(n_splits=args.n_nets, test_size=args.test_size, random_state=args.seed)

    nets = [Ensemble.get_model(
        input_shape=x_train[0][:, :, None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
        n_outputs=y_train[0].shape[0],
        optimizer=args.optimizer,
        dropout=args.dropout
    ) for _ in range(args.n_nets)]

    for i, (train_index, test_index) in enumerate(rs.split(x_train, y_train)):
        x_train_block, y_train_block = x_train[train_index], y_train[train_index]
        x_test_block, y_test_block = x_train[test_index], y_train[test_index]

        history = nets[i].fit(
            x_train_block, y_train_block,
            epochs=args.epochs,
            verbose=args.verbose,
            validation_split=args.val_split,
            callbacks=args.callbacks
        )

        print(f"Evaluation of network {i}:")
        _, acc = nets[i].evaluate(x_test_block, y_test_block, verbose=args.verbose)

    preds = np.array([net.predict(x_test) for net in nets]).mean(axis=0).argmax(axis=1)
    targets = y_test.argmax(axis=1)
    accuracy = np.sum(preds == targets)/preds.size
    print(f"Ensemble accuracy using mean: {accuracy}")


@dataclass
class Args:
    epochs: int = 5
    verbose: int = 0
    val_split: float = 0.2
    n_nets: int = 10
    seed: int = 42
    dropout: bool = True
    test_size: float = 0.2
    # callbacks = [custom_callbacks.get_callbacks()]
    callbacks = []

    optimizer: optimizers.Optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.8)


if __name__ == "__main__":
    # main(Args())
    ensemble_main(Args())

