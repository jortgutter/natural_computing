# import keras.saving.saved_model.model_serialization
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets, optimizers
from dataclasses import dataclass
import sklearn.model_selection
import numpy as np
import time
from itertools import combinations

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


def load_ensemble_data(drop_classes):
    # load data
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # normalize pixel values to range(0,1)
    x_train, x_test = x_train/255.0, x_test/255.0

    # get number of classes from data
    n_classes = len(np.unique(y_test))

    # get list of class labels from data
    classes = np.array(np.unique(y_test))

    # get a list of all possible combinations of classes when dropping drop_classes of them
    class_combos = np.array(list(combinations(classes, len(classes) - drop_classes)))
    # shuffle the lists
    class_combos_perm = np.random.permutation(class_combos)

    # get number of ensemble members needed from amount of combinations
    n_nets = len(class_combos)

    # convert labels to categorical
    y_train_cat = to_categorical(y_train)

    all_datas=[]

    for combo in class_combos_perm:
        # filter training data and labels to only contain the wanted classes
        # get a boolean array of which samples to keep
        indexes = np.isin(y_train, combo)

        # change shape from (x,1) to (x,)
        indexes = np.squeeze(indexes)

        # filter out relevant data and save as dict in list
        all_datas.append({
            'X': x_train[indexes],
            'y': y_train_cat[indexes]
        })

    return n_nets, all_datas  # data has all combinations of data that can be used by the nets

def ensemble_main(args):
    n_nets, all_data = load_ensemble_data(drop_classes=2)  # TODO: put drop_classes in args

    rs = StratifiedShuffleSplit(n_splits=args.n_nets, test_size=args.test_size, random_state=args.seed)  # TODO: smarter splits

    nets = [Ensemble.get_model(  # TODO: unpack all_data per net
        input_shape=x_train[0][:, :, None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
        n_outputs=y_train[0].shape[0],
        optimizer=args.optimizer,
        dropout=args.dropout
    ) for _ in range(args.n_nets)]

    nets[0].summary()

    hists = []

    t = time.time()
    for i, (train_index, test_index) in enumerate(rs.split(x_train, y_train)):
        x_train_block, y_train_block = x_train[train_index], y_train[train_index]
        x_test_block, y_test_block = x_train[test_index], y_train[test_index]

        hists.append(nets[i].fit(
            x_train_block, y_train_block,
            epochs=args.epochs,
            verbose=args.verbose,
            validation_split=args.val_split,
            callbacks=args.callbacks
        ))

        print(f"Evaluation of network {i}:")
        _, acc = nets[i].evaluate(x_test_block, y_test_block, verbose=args.verbose)

    print(f"Training time: {time.time() - t:.2f} seconds")

    preds = np.array([net.predict(x_test) for net in nets]).sum(axis=0).argmax(axis=1)
    targets = y_test.argmax(axis=1)
    accuracy = np.sum(preds == targets)/preds.size
    print(f"Ensemble accuracy using majority voting: {accuracy}")


@dataclass
class Args:
    epochs: int = 2
    verbose: int = 1
    val_split: float = 0.2
    n_nets: int = 20
    seed: int = 42
    dropout: bool = True
    test_size: float = 0.2
    # callbacks = [custom_callbacks.get_callbacks()]
    callbacks = []

    optimizer: optimizers.Optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.8)


if __name__ == "__main__":
    # main(Args())
    ensemble_main(Args())

