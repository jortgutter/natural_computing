import numpy as np
import matplotlib.pyplot as plt


def plot_time(ns, models):
    for ts, net_structure in models:
        plt.scatter(ns, ts, label=net_structure)
    plt.title(f"Training time as function of ensemble size")
    plt.xlabel("Ensemble size")
    plt.ylabel("Runtime (sec)")
    plt.xticks(range(1, max(ns) + 1))
    plt.legend()
    plt.show()


def plot_accs(ns, models):
    plt.figure(figsize=(10, 7))
    for (p_med, p_maj, c_maj, net_structure), col in zip(models, ["tab:blue", "tab:orange"]):
        plt.plot(ns, p_med, color=col, linestyle="solid", label=f'{net_structure} - Prob. median')
        plt.plot(ns, p_maj, color=col, linestyle="dashed", label=f'{net_structure} - Prob. majority')
        plt.plot(ns, c_maj, color=col, linestyle="dotted", label=f'{net_structure} - Class majority')

    plt.title(f"Ensemble test accuracy as function of ensemble size\nusing different voting methods")
    plt.xlabel("Ensemble size")
    plt.ylabel("Ensemble accuracy")
    plt.xticks(range(1, max(ns) + 1))
    plt.legend()
    plt.show()


ns = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
ts4 = [
    132.23,
    255.19,
    397.01,
    531.55,
    656.87,
    1315.36,
    1997.40,
    2694.25,
    3430.91,
    3870.67
]
p_med4 = [
    0.5724,
    0.6792,
    0.675,
    0.6939,
    0.7061,
    0.7653,
    0.7813,
    0.7919,
    0.7865,
    0.7947
]
p_maj4 = [
    0.5724,
    0.6792,
    0.6915,
    0.6962,
    0.7113,
    0.769,
    0.7859,
    0.7938,
    0.7881,
    0.7966
]
c_maj4 = [
    0.5724,
    0.5985,
    0.6747,
    0.6963,
    0.6968,
    0.7658,
    0.7781,
    0.79,
    0.7853,
    0.7924
]

net_struct4 = "5 conv blocks, 4 start channels, 10 epochs"
mod4 = [p_med4, p_maj4, c_maj4, net_struct4]

ts5 = [
    165.52,
    324.16,
    491.97,
    660.36,
    820.80,
    1643.31,
    2481.24,
    3212.09,
    4083.27,
    4880.51
]
p_med5 = [
    0.5726,
    0.7118,
    0.6905,
    0.704,
    0.699,
    0.7739,
    0.7872,
    0.7978,
    0.7981,
    0.8011
]
p_maj5 = [
    0.5726,
    0.7118,
    0.7101,
    0.7061,
    0.7044,
    0.7754,
    0.7927,
    0.7966,
    0.801,
    0.8007
]
c_maj5 = [
    0.5726,
    0.6101,
    0.7074,
    0.7069,
    0.6981,
    0.7702,
    0.7852,
    0.7959,
    0.797,
    0.7986
]

net_struct5 = "5 conv blocks, 5 start channels, 10 epochs"
mod5 = [p_med5, p_maj5, c_maj5, net_struct5]

plot_time(ns, [(ts4, net_struct4), (ts5, net_struct5)])
plot_accs(ns, [mod4, mod5])
