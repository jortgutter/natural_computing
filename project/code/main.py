import numpy as np
from tensorflow.keras import datasets, layers, models



def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    print(len(x_train))

def network():
    network = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Conv2D()
    ])

    return network

def main():
    x = np.array([])
    load_data()


if __name__ == "__main__":
    main()
