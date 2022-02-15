import matplotlib.pyplot as plt
import numpy as np

increase_only = False

N_GENES = 100
MUTATION = 1/N_GENES
N_ITER = 1500
N_TRIALS = 1000

fitnesses = np.zeros((N_TRIALS, N_ITER+1))

x = np.random.randint(2, size=(N_TRIALS, N_GENES))
print(x.shape)
fitnesses[:,0] = x.sum(axis=1)
for n in range(N_ITER):
    # Get array of True and False, where True means mutation
    mutate = (np.random.uniform(low=0, high=1, size=(N_TRIALS, N_GENES)) < MUTATION)

    # XOR with x to flip mutated genes
    x_m = np.logical_xor(x.copy(), mutate)

    if increase_only:
        # TODO: Make more efficient by removing loop
        for t in range(N_TRIALS):
            if x_m[t].sum() > x[t].sum():
                x[t] = x_m[t]
    else:
        x = x_m

    fitnesses[:,n+1] = np.maximum(x.sum(axis=1), fitnesses[:,n])

reached_100 = (fitnesses[:,-1] == 100).sum()

for t in range(N_TRIALS):
    plt.plot(fitnesses[t], color='r', alpha=0.01)
plt.title(f"Fitness over time\n{reached_100}/{N_TRIALS} trials reached 100")
plt.xlabel("Iteration")
plt.ylabel("Fitness (number of 1s)")
plt.show()
