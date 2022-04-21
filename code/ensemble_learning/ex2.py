import matplotlib.pyplot as plt
from math import factorial
import numpy as np


def choose(n: int, k: int) -> int:
    return factorial(n)/(factorial(k) * factorial(n-k))


cs = np.arange(20)
ps = np.arange(1,10)/10

for p in ps:
    probs = np.zeros(len(cs))
    for i, c in enumerate(cs):
        probs[i] = 1- sum([choose(c//2, k) * np.power(p, k) * np.power(1-p, c//2-k) for k in range(c//2)])  
    plt.plot(cs, probs, label=f"p = {p}")

# Is this formula actually the correct one? Somehow adding more people for majority vote makes prediction worse in all cases
plt.title(f"Probability of correct diagnosis with majority vote\nfor different values for c (group size) and p (prob. of individual being correct)")
plt.xlabel("c")
plt.ylabel("p")
plt.legend()
plt.show()
