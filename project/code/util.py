from tensorflow.keras import datasets, layers, models, optimizers
import os


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

    if args.n_decision > 1:
        n_features = args.start_channels * 2 ** (args.n_conv - 1) * (32 / (2 ** (args.n_conv - 2))) ** 2
        for n in range(args.n_decision-1):
            layer_size = n_features / (2 ** (n+1))
            if layer_size <= 10:
                break
            ls.append(layers.Dense(layer_size, activation=args.activation))

    ls.append(layers.Dense(n_outputs, activation='softmax'))

    m = models.Sequential(ls, name=name)

    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def net_summary(x, out_file):
    print(x)

    with open(os.path.join('../out', out_file), 'a') as file:
        file.write(x)
        file.write("\n")
