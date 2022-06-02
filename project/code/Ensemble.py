from tensorflow.keras import datasets, layers, models, optimizers


def get_model(input_shape, n_outputs, optimizer, dropout=False, n=5):

    # models = [single_network(input_shape, n_outputs, optimizer) for _ in range(n)]

    return single_network(input_shape, n_outputs, optimizer, dropout=dropout)


def single_network(input_shape, n_outputs, optimizer, dropout=False) -> models.Sequential:
    m = models.Sequential([l for l in [
        layers.Input(shape=input_shape),
        layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
        layers.Dropout(.5),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.Dropout(.5),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Dropout(.5),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Dropout(.5),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(n_outputs, activation='softmax')
    ] if dropout or type(l) is not layers.Dropout])

    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m
