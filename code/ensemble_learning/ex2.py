import matplotlib.pyplot as plt
# from math import factorial
import numpy as np
from scipy.stats import binom

c_max = 101

cs = np.arange(1, c_max)
ps = np.arange(1, 10) / 10

for p in ps:
    probs = np.zeros(len(cs))
    for i, c in enumerate(cs):
        probs[i] = 1 - binom.cdf(c // 2, c, p)
        # print(f'p: {p} c: {c}, {c // 2}, {probs[i]}')
    plt.plot(cs, probs, label=f"p = {p}")

plt.title(f"Probability of correct diagnosis with majority vote\n"
          f"for different values for c (group size) and p (prob. of individual being correct)")
plt.xlabel("c")
plt.ylabel("p(correct)")
plt.xticks(range(0, c_max, 5))
plt.legend()
plt.show()
