from tensorflow.keras import datasets, optimizers
from dataclasses import dataclass
import numpy as np
import argparse

# Manually defined models
from BaseCNN import BaseCNN
from Ensemble import Ensemble


@dataclass
class Args:
    models = {
        'ensemble': Ensemble,
        'base': BaseCNN
    }
    ensemble_method: str = 'dropout'
    model: str = 'ensemble'
    n_conv: int = 4  # Number of convolutional blocks
    n_decision: int = 2  # Number of dense decision layers
    start_channels: int = 32
    epochs: int = 6
    activation: str = 'relu'
    verbose: int = 1
    val_split: float = 0.1
    n_nets: int = 20  # Max number of nets we want to use
    seed: int = 42
    dropout: bool = True
    p_dropout: float = 0.3
    test_split: float = 0.2
    drop_classes: int = 2
    # callbacks = [custom_callbacks.get_callbacks()]
    callbacks = []

    optimizer: optimizers.Optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.8)


def load_data(args: Args):
    # (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    size_in = y_train.shape[0]
    cutoff = int(args.val_split*size_in)
    indices = np.random.permutation(size_in)

    x_val = x_train[indices[:cutoff]]
    x_train = x_train[indices[cutoff:]]

    y_val = y_train[indices[:cutoff]]
    y_train = y_train[indices[cutoff:]]

    return x_train, y_train, x_val, y_val,  x_test, y_test


def main(args):
    data = load_data(args)
    model = args.models[args.model](args)
    model.train(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network (ensemble) on CIFAR10 data and predict')
    parser.add_argument('model', help='Either \'base\' or \'ensemble\'')
    parser.add_argument('epochs', type=int, help='Number of epochs for training')

    parser.add_argument('n_conv', type=int, help='Number of convolution blocks in the network')
    parser.add_argument('start_channels', type=int, help='Number of output channels of first convolution block (after that gets doubled every block)')

    parser.add_argument('--n_decision', metavar='ND', type=int, default=1, help='Number of dense decision layers of the network')
    parser.add_argument('--activation', metavar='A', type=str, default='relu', help='Activation function for convolution blocks')
    parser.add_argument('--n_nets', metavar='N', type=int, default=5, help='Number of networks in ensemble')
    parser.add_argument('--ensemble_method', metavar='M', type=str, default='dropout', help='Method of distributing data to ensemble')
    parser.add_argument('--drop_classes', metavar='DC', type=int, default=2, help='Number of classes to drop for ensemble data distribution (dropout)')
    parser.add_argument('--val_split', metavar='VS', type=float, default=0.1, help='Fraction of data used for validation')
    parser.add_argument('--test_split', metavar='TS', type=float, default=0.2, help='Fraction of data used for testing')
    parser.add_argument('--dropout', metavar='DR', action='store_const', const=True, default=False, help='Activate dropout during training')
    parser.add_argument('--p_dropout', metavar="PD", type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--silent', dest='verbose', metavar='SI', action='store_const', const=False, default=True, help='Deactivate verbosity')
    parser.add_argument('--seed', metavar='SE', type=int, default=42, help='Seed used for randomness')

    args: argparse.Namespace = parser.parse_args()
    main(Args(**vars(args)))  # The Args class is probably not necessary anymore with the argparser...
