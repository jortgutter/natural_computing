import numpy as np
import tensorflow as tf


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print(len(x_train))

def main():
    load_data()

if __name__ == "__main__":
    main()