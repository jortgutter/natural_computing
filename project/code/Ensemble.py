from tensorflow.keras import datasets, layers, models, optimizers
from keras.utils.np_utils import to_categorical
import numpy as np
from itertools import combinations
import sklearn.model_selection
import time
import sys


class Ensemble:
    def __init__(self, args):
        self.args = args
        self.method = self.args.ensemble_method

    def single_network(self, input_shape, n_outputs, optimizer, dropout=False, name=""):
        m = models.Sequential(
            layers=[l for l in [
                layers.Input(shape=input_shape),
                layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
                #layers.Dropout(.5),
                # layers.MaxPooling2D((2, 2)),
                layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
                #layers.Dropout(.5),
                # layers.MaxPooling2D((2, 2)),
                layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
                layers.Dropout(.5),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.Dropout(.5),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(n_outputs, activation='softmax')
            ] if dropout or type(l) is not layers.Dropout],
            name=name
        )

        m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return m

    def create_class_combos(self):
        # get list of class labels from data
        classes = np.unique(self.y_train)

        # get number of classes from data
        n_classes = classes.size

        # get a list of all possible combinations of classes when dropping drop_classes of them
        self.class_combos = np.array(list(combinations(classes, n_classes - self.args.drop_classes)))

    def get_data_portion(self, idx):
        # get indices of relevant data
        index_bools = np.isin(self.y_train, self.class_combos[idx]).squeeze()
        x_train_selected = self.x_train[index_bools]
        y_train_selected = self.y_train_cat[index_bools]
        return x_train_selected, y_train_selected

    #@profile
    def train(self, data):
        self.train_prep(data)

        self.nets = []

        hists = []

        t = time.time()

        self.n_nets = 2
        for i in range(self.n_nets):
            print(f'Start training of model {i+1}/{self.n_nets}')
            x_train, y_train = self.get_data_portion(i)

            net = self.single_network(
                input_shape=x_train[0][:, :, None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
                n_outputs=y_train[0].shape[0],
                optimizer=self.args.optimizer,
                dropout=self.args.dropout,
                name=f"Model_{i}_of_{self.n_nets}"
            )

            self.nets.append(net)

            if i == 0:
                net.summary()

            self.nets[i].fit(
                x_train, y_train,
                epochs=self.args.epochs,
                verbose=self.args.verbose,
                validation_data=(self.x_val, self.y_val_cat),
                callbacks=self.args.callbacks
            )

        print(f"Training time: {time.time() - t:.2f} seconds")

        preds = np.array([net.predict(self.x_test) for net in self.nets]).sum(axis=0).argmax(axis=1)
        targets = self.y_test_cat.argmax(axis=1)
        accuracy = np.sum(preds == targets) / preds.size
        print(f"Ensemble accuracy using majority voting: {accuracy}")

    def train_prep(self, data):
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test_cat = data

        self.y_train_cat = to_categorical(self.y_train)
        self.y_val_cat = to_categorical(self.y_val)
        self.y_test_cat = to_categorical(self.y_test_cat)

        if self.method == 'dropout':
            # get all possible combinations of classes
            self.create_class_combos()

            # get number of ensemble members needed from amount of combinations
            self.n_nets = len(self.class_combos)
        else:
            self.n_nets = self.args.n_nets
