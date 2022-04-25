from scipy.special import comb as choose
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np
import math

ws = np.arange(1, 100) / 100

probs = np.zeros(len(ws))
for i, w in enumerate(ws):
    probs[i] = sum([choose(10, n) * np.power(0.6, n) * np.power(1 - 0.6, 10 - n) * 0.8 for n in range(math.floor(5 - 5 * w / (1 - w)) + 1, 10)]) + \
               sum([choose(10, n) * np.power(0.6, n) * np.power(1 - 0.6, 10 - n) * 0.2 for n in range(math.floor(5 + 5 * w / (1 - w)) + 1, 10)])
plt.plot(ws, probs)
plt.title(f"Probability of correct diagnosis with majority vote\n"
          f"for different values for $\omega$ (strong classifier weight)")
plt.xlabel("$\omega$")
plt.ylabel("p(correct)")
plt.xticks(np.arange(11)/10)
plt.ylim((0.7, 0.9))
plt.show()


# 3d
# ==========================================================

errors = np.arange(1, 100)/100
weights = np.array([math.log((1-e)/e) for e in errors])

plt.plot(errors, weights)
plt.vlines(0.5, weights.min(), weights.max(), color='k', alpha=0.3)
plt.hlines(0, 0, 1, color='k', alpha=0.3)
plt.title("Weight assigned to base-learner for different error values (e)")
plt.xlabel("e")
plt.ylabel("Weight")
plt.show()
