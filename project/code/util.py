from tensorflow.keras import datasets, layers, models, optimizers


def get_network(input_shape, n_outputs, optimizer, args, name=None) -> models.Sequential:
    ls = [layers.Input(shape=input_shape)]
    for n in range(args.n_conv):
        ls.append(
            layers.Conv2D(args.start_channels * 2 ** n, (3, 3), activation=args.activation, padding='same'))
        if n >= 2:
            if args.dropout:
                ls.append(layers.Dropout(args.p_dropout))
            ls.append(layers.MaxPooling2D((2, 2)))

    ls.append(layers.Flatten())
    ls.append(layers.Dense(n_outputs, activation='softmax'))
    m = models.Sequential(ls, name=name)

    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m