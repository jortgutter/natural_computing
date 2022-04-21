import matplotlib.pyplot as plt
#from math import factorial
import numpy as np
from scipy.stats import binom


#def choose(n: int, k: int) -> int:
#   return factorial(n)/(factorial(k) * factorial(n-k))


cs = np.arange(1,21)
ps = np.arange(1,10)/10

for p in ps:
    probs = np.zeros(len(cs))
    for i, c in enumerate(cs):
        #probs[i] = 1- sum([choose(c//2, k) * np.power(p, k) * np.power(1-p, c//2-k) for k in range(c//2)])
       
        
        probs[i] = 1 - binom.cdf(c//2,c,p)
        print(f'p: {p} c: {c}, {c//2}, {1 - binom.cdf(c//2,c,p)}')
    plt.plot(cs, probs, label=f"p = {p}")

# Is this formula actually the correct one? Somehow adding more people for majority vote makes prediction worse in all cases
plt.title(f"Probability of correct diagnosis with majority vote\nfor different values for c (group size) and p (prob. of individual being correct)")
plt.xlabel("c")
plt.ylabel("p")
plt.xticks(cs)
plt.yticks(ps)
plt.legend()
plt.show()
