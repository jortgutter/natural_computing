from keras.utils.np_utils import to_categorical
import tensorflow as tf
import util
import time
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class BaseCNN:
    def __init__(self, args):
        self.args = args
        tf.random.set_seed = self.args.seed

    def set_model(self, input_shape, n_outputs, optimizer):
        self.model = util.get_network(input_shape, n_outputs, optimizer, self.args)

    def train(self, data):
        x_train, y_train, x_val, y_val, x_test, y_test = data

        y_train = to_categorical(y_train)
        y_val   = to_categorical(y_val)
        y_test  = to_categorical(y_test)

        # Change to Ensemble.get_model() for other model
        self.set_model(
            input_shape=x_train[0][:, :, None].shape if len(x_train[0].shape) == 2 else x_train[0].shape,
            n_outputs=y_train[0].shape[0],
            optimizer=self.args.optimizer
        )

        self.model.summary(print_fn=lambda x: util.net_summary(x, self.args.output_filename))

        t = time.time()
        hist = self.model.fit(
            x_train, y_train,
            epochs=self.args.epochs,
            verbose=self.args.verbose,
            validation_data=(x_val, y_val),
            callbacks=self.args.callbacks
        )

        t_train = time.time() - t

        print(f"Training time: {t_train} seconds")

        print("=================================================================")
        _, acc = self.model.evaluate(x_test, y_test, verbose=self.args.verbose)

        with open(os.path.join('../out', self.args.output_filename), 'a') as file:
            file.write("\n".join([f"{k}: {v}" for k, v in hist.history.items()]) + "\n\n")
            file.write(f"Training time: {t_train:.2f} seconds\n")
            file.write(f"accuracy: {acc}")

