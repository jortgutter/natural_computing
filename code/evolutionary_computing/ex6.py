import math
import random
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Route:
    order: np.ndarray
    score: float = 0

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.score == other.score


# load the data:
locations = [[float(c) for c in x.split()] for x in open('file-tsp.txt', 'r').read().split('\n')]
# save number of locations:
n = len(locations)
print(f'Loaded {len(locations)} locations successfully!')


# calculate the distance between two locations
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def generate_population(p):
    population = np.array([Route(order=np.random.permutation(n)) for i in range(p)])
    for candidate in population:
        candidate.score = evaluate(candidate.order)
    return population


# CORRECT
def binary_tournament(pop):
    selection = np.random.choice(pop, 2)
    return selection[0] if selection[0].score >= selection[1].score else selection[1]


def local_search(candidate):
    pass


def crossover(parents):
    if len(parents) != 2:
        print('ERROR: not enough or too many parents!')
        exit(1)
    # generate two random indices:
    indices = np.sort(np.random.randint(n, size=2))
    # create child arrays and create slices:
    children = [np.zeros(n).astype(int) - 1 for i in range(2)]
    slices = [parents[i].order[indices[0]:indices[1]] for i in range(2)]

    # Cross over slices
    for i in range(2):
        children[i][indices[0]:indices[1]] = slices[i]

    # Fill child arrays with values of opposite parent:
    for i in range(2):
        # keep track of current location within cross-parent:
        p_i = 0
        # Loop over empty child values:
        for c_i in range(n):
            if children[i][c_i] == -1:
                # look for next cross-parent value:
                while parents[1 - i].order[p_i] in slices[i]:
                    p_i += 1
                # fill in cross-parent value:
                children[i][c_i] = parents[1 - i].order[p_i]
                p_i += 1
    # create data classes for children;
    return np.array([Route(order=children[0]), Route(order=children[1])])


def mutate(candidate, p):
    for i in range(n):
        if np.random.random() < p:
            placeholder = candidate.order[i]
            swap_i = np.random.randint(n)
            candidate.order[i] = candidate.order[swap_i]
            candidate.order[swap_i] = placeholder


def evaluate(order):
    dist = np.sum([distance(locations[order[i]], locations[order[i + 1]]) for i in range(len(order) - 1)])
    return 1 / dist


def darwin(locations, pop_n, mutate_p, its, runs=1, use_ma=False):
    best_results = []
    avg_results = []
    for r in range(runs):
        print(f'Starting run {r}')
        # generate initial population:
        population = generate_population(pop_n)
        best_run_results = []
        avg_run_results = []
        for it in range(its):
            # select n/2 parents:
            parent_pairs = []
            for i in range(int(pop_n / 2)):
                parent_pairs.append([binary_tournament(population), binary_tournament(population)])

            # generate offspring (crossover):
            children = np.array([])
            for pair in parent_pairs:
                children = np.append(children, crossover(pair))

            # mutate offspring:
            for child in children:
                mutate(child, p)

            # if MA: perform local search:
            if use_ma:
                children = [local_search(child) for child in children]

            # evaluate offspring:
            for child in children:
                child.score = evaluate(child.order)

            # select next generation by keeping the best pop_n individuals:
            gene_pool = np.append(population, children)
            gene_pool.sort()
            population = gene_pool[-pop_n:]

            # save current best scoring individual:
            best_run_results.append(population[-1].score)
            avg_run_results.append(np.mean([p.score for p in population]))

        best_results.append(best_run_results)
        avg_results.append(avg_run_results)

    return best_results, avg_results


# parameters
n_it = 1500
n_runs = 10
population = 20
p = 3 / n
# evolution:
best_results_EA, avg_results_EA = darwin(locations[:10], pop_n=population, mutate_p=p, its=n_it, runs=n_runs, use_ma=False)

for res in best_results_EA:
    plt.plot(res, c='red')
for res in avg_results_EA:
    plt.plot(res, c='blue')

plt.show()


