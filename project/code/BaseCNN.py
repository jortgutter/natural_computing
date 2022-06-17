from tensorflow.keras import datasets, layers, models, optimizers
from keras.utils.np_utils import to_categorical
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class BaseCNN:
    def __init__(self, args):
        self.args = args

    def set_model(self, input_shape, n_outputs, optimizer, dropout=False):
        m = models.Sequential([l for l in [
            layers.Input(shape=input_shape),
            layers.Conv2D(48, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.3),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.3),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(212, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.3),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(424, (3, 3), activation='relu', padding='same'),
            layers.Dropout(.3),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(n_outputs, activation='softmax')
        ] if dropout or type(l) is not layers.Dropout])

        m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = m

    def load_model(self, path):
        pass

    def train(self, data):
        x_train, y_train, x_val, y_val, x_test, y_test = data

        y_train = to_categorical(y_train)
        y_val   = to_categorical(y_val)
        y_test  = to_categorical(y_test)

        # Change to Ensemble.get_model() for other model
        self.set_model(
            input_shape=x_train[0][:, :, None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
            n_outputs=y_train[0].shape[0],
            optimizer=self.args.optimizer,
            dropout=True
        )

        print(self.model.summary())

        print(x_train.shape)
        print(y_train)
        history = self.model.fit(
            x_train, y_train,
            epochs=self.args.epochs,
            verbose=self.args.verbose,
            validation_data=(x_val, y_val),
            callbacks=self.args.callbacks
        )

        print("=================================================================")
        _, acc = self.model.evaluate(x_test, y_test, verbose=self.args.verbose)
