# import keras.saving.saved_model.model_serialization
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets

# Manually defined models
import BaseCNN
import Ensemble


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    # normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.2)

    print("=================================================================")
    _, acc = model.evaluate(x_test, y_test, verbose=1)


def main():
    x_train, y_train, x_test, y_test = load_data()

    model = BaseCNN.get_model()  # Change to Ensemble.get_model() for other model
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
