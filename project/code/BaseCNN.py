from tensorflow.keras import datasets, layers, models, optimizers


def get_model(input_shape, n_outputs, optimizer) -> models.Sequential:
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(n_outputs, activation='softmax')
    ])

    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def load_model(path):
    pass
