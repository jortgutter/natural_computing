from tensorflow.keras import datasets, layers, models, optimizers


def get_model(input_shape, n_outputs, optimizer, dropout=False) -> models.Sequential:
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
    return m


def load_model(path):
    pass
