# import keras.saving.saved_model.model_serialization
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras import datasets, optimizers
from dataclasses import dataclass
import sklearn.model_selection
import numpy as np
import time
import sys


# Manually defined models
from BaseCNN import BaseCNN
from Ensemble import Ensemble


import custom_callbacks


def load_data():
    # (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test




def main(args):
    x_train, y_train, x_test, y_test = load_data()

    model = args.models[args.use_model](args)
    model.fit(x_train, y_train, x_test, y_test)










@dataclass
class Args:
    models={
        'Ensemble': Ensemble,
        'Base': BaseCNN
    }
    use_model='Ensemble'
    epochs: int = 2
    verbose: int = 1
    val_split: float = 0.2
    n_nets: int = 20
    seed: int = 42
    dropout: bool = True
    test_size: float = 0.2
    drop_classes: int = 2
    # callbacks = [custom_callbacks.get_callbacks()]
    callbacks = []

    optimizer: optimizers.Optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.8)


if __name__ == "__main__":
    main(Args())
    # ensemble_main(Args())

