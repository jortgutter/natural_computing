# import keras.saving.saved_model.model_serialization
from keras.utils.np_utils import to_categorical
from IPython.display import clear_output
from tensorflow.keras import datasets
from matplotlib import pyplot as plt
import keras

# Manually defined models
import BaseCNN
import Ensemble

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()

callbacks_list = [PlotLearning()]

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
    history = model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.2, callbacks=callbacks_list)

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
