from tensorflow.keras import datasets, layers, models, optimizers
from keras.utils.np_utils import to_categorical
import numpy as np
from itertools import combinations
import sklearn.model_selection
import time
import sys


class Ensemble():
    def __init__(self, args):
        self.args = args
        self.method = self.args.ensemble_method
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def get_model(self, input_shape, n_outputs, optimizer, dropout=False, n=5):

        # models = [single_network(input_shape, n_outputs, optimizer) for _ in range(n)]

        return self.single_network(input_shape, n_outputs, optimizer, dropout=dropout)

    def single_network(self, input_shape, n_outputs, optimizer, dropout=False):
        m = models.Sequential([l for l in [
            layers.Input(shape=input_shape),
            layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.5),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.5),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.5),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.5),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(n_outputs, activation='softmax')
        ] if dropout or type(l) is not layers.Dropout])

        m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return m

    def create_class_combos(self, drop_classes: int):
        # get number of classes from data
        n_classes = len(np.unique(self.y_test))

        # get list of class labels from data
        classes = np.array(np.unique(self.y_test))

        # get a list of all possible combinations of classes when dropping drop_classes of them
        self.class_combos = np.array(list(combinations(classes, n_classes - drop_classes)))

    def get_data_portion(self, idx):
        # get indices of relevant data
        index_bools = np.isin(self.y_train, self.class_combos[idx]).squeeze()
        print(f'size of bools: {index_bools.shape}')
        x_train_selected = self.x_train[index_bools]
        y_train_selected = self.y_train_cat[index_bools]
        return x_train_selected, y_train_selected

    def train_ensemble(self):
        self.nets = []

        hists = []

        t = time.time()
        for i in range(self.n_nets):
            x_train, y_train = self.get_data_portion(i)

            net = self.get_model(
                input_shape=x_train[0][:, :, None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
                n_outputs=y_train[0].shape[0],
                optimizer=self.args.optimizer,
                dropout=self.args.dropout
            )

            self.nets.append(net)

            if i == 0:
                net.summary()

            hists.append(self.nets[i].fit(
                x_train, y_train,
                epochs=self.args.epochs,
                verbose=self.args.verbose,
                validation_split=self.args.val_split,
                callbacks=self.args.callbacks
            ))

            # TODO: Fix rest of this function!!
            print(f"Evaluation of network {i}:")
            _, acc = self.nets[i].evaluate(self.x_test, self.y_test_cat, verbose=self.args.verbose)

        print(f"Training time: {time.time() - t:.2f} seconds")

        preds = np.array([net.predict(self.x_test) for net in self.nets]).sum(axis=0).argmax(axis=1)
        targets = self.y_test.argmax(axis=1)
        accuracy = np.sum(preds == targets) / preds.size
        print(f"Ensemble accuracy using majority voting: {accuracy}")

    def fit(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_train_cat = to_categorical(self.y_train)
        self.y_test = y_test
        self.y_test_cat = to_categorical(self.y_test)

        if self.method == 'dropout':
            # get all possible combinations of classes
            self.create_class_combos(drop_classes=self.args.drop_classes)

            # get number of ensemble members needed from amount of combinations
            self.n_nets = len(self.class_combos)
        else:
            self.n_nets = self.args.n_nets

        # train ensemble
        self.train_ensemble()
