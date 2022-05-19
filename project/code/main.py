# import keras.saving.saved_model.model_serialization
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets, optimizers

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
        input_shape=x_train[0].shape,
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


if __name__ == "__main__":
    main()
