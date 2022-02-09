import math
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class Route:
    order:[]
    score:0


# load the data:
locations = [[float(c) for c in x.split()] for x in open('file-tsp.txt', 'r').read().split('\n')]
# save number of locations:
n = len(locations)
print(f'Loaded {len(locations)} locations successfully!')

# calculate the distance between two locations
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def generate_population(p):
    return [np.random.permutation(i) for i in range(p)]

def binary_tournament(pop):
    selection = random.sample(pop, 2)
    return selection[0] if selection[0].score >= selection[1] else selection[1]

def local_search(candidate):
    pass

def crossover(parents):
    pass

def mutate(candidate):
    return


def evaluate(order):
    dist = sum([distance(locations[order[i]], locations[order[i+1]])] for i in range(len(order)-1))
    return 1/dist

def darwin(locations, pop_n, mutate_p, k, its, runs=1, use_MA=False):
    n = len(locations)
    results = []
    for r in range(runs):
        # generate initial population:
        population = generate_population(pop_n)
        run_results = []
        for it in range(its):
            # select n/2 parents:
            parent_pairs = []
            for i in range(int(pop_n/2)):
                parent_pairs.append([binary_tournament(population), binary_tournament(population)])

            # generate offspring (crossover):
            children = []
            for pair in parent_pairs:
                children += crossover(pair)

            # mutate offspring:
            children = map(lambda x: mutate(x), children)

            # if MA: perform local search:
            if use_MA:
                children = [local_search(child) for child in children]

            # evaluate offspring:
            for child in children:
                child.score = evaluate(child.order)

            # select next generation by keeping the best pop_n individuals:
            gene_pool = population + children
            gene_pool.sort(key=lambda x: x.score)
            pop = gene_pool[:pop_n]

            # save current best scoring individual:
            run_results.append(gene_pool[0].score)



        # generate random candidate:
        x = np.random.permutation(n)
        init_route = Route(order=x, score=evaluate(x))
        # iterate over number of generations:
        for i in range(its):

            pass
        results.append(run_results)

    return results

# parameters
n_it = 1500
n_runs = 10
population=50
# evolution:
results_EA = darwin(locations, its=n_it, runs=n_runs, use_MA=False)
print(results_EA)



