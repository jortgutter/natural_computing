import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets, layers, models, optimizers

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    # normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test

def define_model():
    network = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(network.summary())
    return network

def train_model(x_train, y_train, x_test, y_test):
    model = define_model()
    model.fit(x_train, y_train, epochs=10, verbose=1)
    print("=================================================================")
    _, acc = model.evaluate(x_test, y_test, verbose=1)




def main():
    x_train, y_train, x_test, y_test = load_data()
    train_model(x_train, y_train, x_test, y_test)



if __name__ == "__main__":
    main()