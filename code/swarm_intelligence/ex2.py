import numpy as np
import matplotlib.pyplot as plt

n_it = 100

omegas = [0.5, 0.7]
alpha = 1.5  # a1, a2 are both 1.5 in both scenarios
rs = [0.5, 1]

positions = [[],[]]

def evaluate(pos):
    return pos**2

for i in range(2):
    x = 20
    v = 10
    best = x
    positions[i].append(x)
    for j in range(n_it):
        # update velocity:
        v = omegas[i]*v + 2*alpha*rs[i]*(best - x)
        print(v)
        # update position:
        x = x + v

        # update best:
        if evaluate(x) < evaluate(best):
            best = x

        # store position for plotting:
        positions[i].append(x)

# plotting:
plt.plot(positions[0], label='case 1')
plt.plot(positions[1], label='case 2')
plt.legend()
plt.show()
