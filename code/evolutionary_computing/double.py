import math
import random
import numpy as np
import copy
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Route:
    order1: np.ndarray
    order2: np.ndarray
    score1: float = 0
    score2: float = 0
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
    population = np.array([Route(order1=np.random.permutation(n), order2=np.random.permutation(n)) for i in range(p)])
    for candidate in population:
        candidate.score1 = evaluate(candidate.order1)
        candidate.score2 = evaluate(candidate.order2)
        candidate.score = max(candidate.score1, candidate.score2)
    return population


# CORRECT
def binary_tournament(pop):
    selection = np.random.choice(pop, 2)
    return max(selection[0], selection[1])


def opt_2_swap(route, i , j):
    # reverse middle bit:
    return np.append(np.append(route[:i], route[i:j][::-1]), route[j:])


def local_search(candidate):
    orig_route = candidate.order
    best_route = orig_route
    best_dist = total_dist(best_route)
    for i in range(0, len(best_route)-1):
        for j in range(i + 2, len(best_route)+1):
            if j-i == 1:
                # Slice is of size 1, swapping will not do anything
                continue
            new_route = copy.deepcopy(orig_route)
            slice = orig_route[i:j]
            new_route[i:j] = slice[::-1]
            new_dist = total_dist(new_route)
            if new_dist < best_dist:
                best_route = new_route
                best_dist = new_dist

    candidate.order = best_route
    candidate.score = evaluate(best_route)


def crossover(parents):
    if len(parents) != 2:
        print('ERROR: not enough or too many parents!')
        exit(1)
    parent_genes = np.array([parents[0].order1, parents[0].order2, parents[0].order1, parents[0].order2])
    np.random.shuffle(parent_genes)
    # generate two sets of random indices:
    indices1 = np.sort(np.random.randint(n, size=2))
    indices2 = np.sort(np.random.randint(n, size=2))
    # create child arrays and create slices:
    children = [np.zeros(n).astype(int) - 1 for i in range(4)]
    slices1 = [parent_genes[i][indices1[0]:indices1[1]] for i in range(2)]
    slices2 = [parent_genes[i+2][indices2[0]:indices2[1]] for i in range(2)]

    # Cross over slices
    for i in range(2):
        children[i][indices1[0]:indices1[1]] = slices1[i]
        children[i+2][indices2[0]:indices2[1]] = slices2[i]

    # Fill child arrays with values of opposite parent:

    for i in range(2):
        # keep track of current location within cross-parent:
        p_i1 = 0
        p_i2 = 0
        # Loop over empty child values:
        for c_i in range(n):
            if children[i][c_i] == -1:
                # look for next cross-parent value:
                while parent_genes[1 - i][p_i1] in slices1[i]:
                    p_i1 += 1

                # fill in cross-parent value:
                children[i][c_i] = parent_genes[1 - i][p_i1]
                p_i1 += 1
        for c_i in range(n):
            if children[i+2][c_i] == -1:
                # look for next cross-parent value:
                while parent_genes[1 - i+2][p_i2] in slices2[i]:
                    p_i2 += 1
                # fill in cross-parent value:
                children[i+2][c_i] = parent_genes[1 - i+2][p_i2]
                p_i2 += 1
    # create data classes for children;
    np.random.shuffle(children)
    return np.array([Route(order1=children[0], order2=children[2]), Route(order1=children[1], order2=children[3])])


def mutate(candidate, p):
    for i in range(n):
        if np.random.random() < p:
            placeholder1 = candidate.order1[i]
            swap_i1 = np.random.randint(n)
            candidate.order1[i] = candidate.order1[swap_i1]
            candidate.order1[swap_i1] = placeholder1
        if np.random.random() < p:
            placeholder2 = candidate.order2[i]
            swap_i2 = np.random.randint(n)
            candidate.order2[i] = candidate.order2[swap_i2]
            candidate.order2[swap_i2] = placeholder2


def total_dist(order):
    return np.sum([distance(locations[order[i]], locations[order[i + 1]]) for i in range(len(order) - 1)])


def evaluate(order):
    return 1 / total_dist(order)


def darwin(locations, pop_n, mutate_p, its, runs=1, use_ma=False):
    best_results = []
    avg_results = []
    for r in range(runs):
        print(f'Starting run {r}')
        # generate initial population:
        population = generate_population(pop_n)
        best_run_results = []
        avg_run_results = []
        for it in tqdm(range(its)):
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
                mutate(child, mutate_p)

            # if MA: perform local search:
            if False:
                for child in children:
                    local_search(child)

            # evaluate offspring:
            for child in children:

                child.score1 = evaluate(child.order1)
                child.score2 = evaluate(child.order2)
                if use_ma:
                    child.score = max(child.score1, child.score2)
                else:
                    child.score = child.score1


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
population = 10
p = 1 / n
# evolution:
best_results_EA, avg_results_EA = darwin(locations[:10], pop_n=population, mutate_p=p, its=n_it, runs=n_runs, use_ma=False)

for res in best_results_EA:
    plt.plot(res, c='red')
for res in avg_results_EA:
    plt.plot(res, c='blue')

plt.show()

best_results_MA, avg_results_MA = darwin(locations[:10], pop_n=population, mutate_p=p, its=n_it, runs=n_runs, use_ma=True)

for res in best_results_EA:
    plt.plot(res, c='red')
for res in avg_results_EA:
    plt.plot(res, c='blue')
for res in best_results_MA:
    plt.plot(res, c='orange')
for res in avg_results_MA:
    plt.plot(res, c='green')

plt.show()


