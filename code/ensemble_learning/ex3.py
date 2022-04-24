from scipy.special import comb as choose
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np
import math

ws = np.arange(1, 100) / 100

probs = np.zeros(len(ws))
for i, w in enumerate(ws):
    probs[i] = sum([choose(10, i) * np.power(0.6, i) * np.power(1 - 0.6, 10 - i) * 0.8 for i in range(5 * math.ceil(5/(1-w)), 11)]) + \
               sum([choose(10, i) * np.power(0.6, i) * np.power(1 - 0.6, 10 - i) * 0.2 for i in range(6 * math.ceil((5-10*w)/(1-w)), 11)])  # TODO: fix calculation with weight w
plt.plot(ws, probs)

plt.title(f"Probability of correct diagnosis with majority vote\n"
          f"for different values for $\omega$ (strong classifier weight)")
plt.xlabel("$\omega$")
plt.ylabel("p")
plt.xticks(np.arange(10)/10)
plt.show()
