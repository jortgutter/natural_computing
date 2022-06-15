from tensorflow.keras import datasets, layers, models, optimizers
from keras.utils.np_utils import to_categorical
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class BaseCNN():
    def __init__(self, args):
        self.args=args
        self.x_train, self.y_train, self.x_test , self.y_test = None, None, None, None


    def get_model(self,input_shape, n_outputs, optimizer, dropout=False):
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
        self.model=m


    def load_model(self, path):
        pass

    def fit(self, x_train, y_train, x_test, y_test):


        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        # Change to Ensemble.get_model() for other model
        self.get_model(
            input_shape=self.x_train[0][:, :, None].shape if len(self.x_train[0].shape) == 2 else self.x_train[0].shape,
            n_outputs=self.y_train[0].shape[0],
            optimizer=self.args.optimizer,
            dropout=True
        )

        print(self.model.summary())

        self.train_model()

    def train_model(self):
        print(self.x_train.shape)
        print(self.y_train)
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.args.epochs,
            verbose=self.args.verbose,
            validation_split=self.args.val_split,
            callbacks=self.args.callbacks
        )

        print("=================================================================")
        _, acc = self.model.evaluate(self.x_test, self.y_test, verbose=self.args.verbose)