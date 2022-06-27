import numpy as np
import matplotlib.pyplot as plt

ns = [1, 2, 3, 4, 5, 10, 15, 20]
ts = [
        132.23,
        255.19,
        397.01,
        531.55,
        656.87,
        1315.36,
        1997.40,
        2694.25
    ]
p_med = [
            0.5724,
            0.6792,
            0.675,
            0.6939,
            0.7061,
            0.7653,
            0.7813,
            0.7919
        ]
p_maj = [
            0.5724,
            0.6792,
            0.6915,
            0.6962,
            0.7113,
            0.769,
            0.7859,
            0.7938
        ]
c_maj = [
            0.5724,
            0.5985,
            0.6747,
            0.6963,
            0.6968,
            0.7658,
            0.7781,
            0.79
        ]

plt.scatter(ns, ts)
plt.title("Runtime as function of ensemble size")
plt.xlabel("Ensemble size")
plt.ylabel("Runtime (sec)")
plt.xticks(range(1, max(ns) + 1))
plt.show()

plt.plot(ns, p_med, label='Prob. median')
plt.plot(ns, p_maj, label='Prob. majority')
plt.plot(ns, c_maj, label='Class majority')
plt.title("Ensemble test accuracy as function of ensemble size\nusing multiple voting methods")
plt.xlabel("Ensemble size")
plt.ylabel("Ensemble accuracy")
plt.xticks(range(1, max(ns) + 1))
plt.show()
