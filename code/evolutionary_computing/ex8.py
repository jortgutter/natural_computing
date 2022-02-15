import matplotlib.pyplot as plt
import numpy as np

from tree import Tree
from tqdm import tqdm
from node import *
import copy




def load_data(filename):
    with open(filename, "r") as file:
        xs = [float(term) for term in file.readline().split(", ")]
        ys = [float(term) for term in file.readline().split(", ")]
    return xs, ys


def get_predictions(tree: Tree, terminal_value: TerminalValue, x_vals: List[float]) -> List[float]:
    results = []

    for x in x_vals:
        terminal_value.v = x
        results.append(tree.root.evaluate())

    return results


def plot_predictions(tree: Tree, xs: List[float], ys: List[float]):
    res = get_predictions(tree, term_val, xs)

    plt.plot(xs, ys, label="True", c="green")
    plt.plot(xs, res, label="Estimated", c="black")
    plt.legend()
    plt.show()


def crossover(tree1: Tree, tree2: Tree):
    # Randomly select nodes from trees
    node1 = np.random.choice(tree1.root.get_reference_list())
    node2 = np.random.choice(tree2.root.get_reference_list())

    # Retrieve Node parents
    parent1 = node1.p
    parent2 = node2.p

    # Update children
    if parent1:
        for i, c1 in enumerate(parent1.children):
            if node1 == c1:
                parent1.children[i] = node2
                break
    else:
        tree1.root = node2

    if parent2:
        for i, c2 in enumerate(parent2.children):
            if node2 == c2:
                parent2.children[i] = node1
                break
    else:
        tree2.root = node1

    node1.update_parent(parent2)
    node2.update_parent(parent1)


def protected_div(a, b):
    return 1 if b == 0 else np.divide(a, b)


def protected_log(a):
    return 0 if a <= 0 else np.log(a)


functions = [
    (np.add, 2),
    (np.subtract, 2),
    (np.multiply, 2),
    (protected_div, 2),
    (protected_log, 1),
    (np.exp, 1),
    (np.sin, 1),
    (np.cos, 1)
]

xs, ys = load_data("data_ex8.txt")
term_val = TerminalValue(0)

MAX_DEPTH = 3
POP_SIZE = 1000
N_GEN = 50
P_CROSSOVER = 0.7
k = 500


def calculate_fitness(tree: Tree, term_val: TerminalValue, xs: List[float], ys: List[float]):
    preds = get_predictions(tree, term_val, xs)

    errors = [abs(y - pred) for y, pred in zip(ys, preds)]

    return -sum(errors)


def select_parents(population: List[Tree], k: int) -> Tuple[Tree, Tree]:
    group1 = np.random.permutation(population)[:k]
    group2 = np.random.permutation(population)[:k]

    return get_best(group1), get_best(group2)


def get_best(group: np.ndarray) -> Tree:
    best_fit = -np.inf
    best_tree = None

    for tree in group:
        if tree.fitness > best_fit:
            best_fit = tree.fitness
            best_tree = tree

    return best_tree


population = [Tree(FunctionNode(None, functions=functions, terminal_value=term_val, max_depth=MAX_DEPTH)) for _ in range(POP_SIZE)]
#for tree in population:
#    plot_predictions(tree, xs, ys)


optimal_fitness = [np.inf]
optimal_tree = None
for tree in population:
    fitness = calculate_fitness(tree, term_val, xs, ys)
    if fitness < optimal_fitness[0]:
        optimal_fitness[0] = fitness
        optimal_tree = tree
    tree.update_fitness(fitness)

for gen in tqdm(range(N_GEN)):
    replaced = 0
    new_pop = []

    for i in range(0, POP_SIZE, 2):
        t1, t2 = select_parents(population, k)

        if np.random.uniform(0, 1) < 0.7:  # Crossover
            o1 = copy.deepcopy(t1)
            o2 = copy.deepcopy(t2)

            crossover(o1, o2)
            o1.update_fitness(calculate_fitness(o1, term_val, xs, ys))
            o2.update_fitness(calculate_fitness(o2, term_val, xs, ys))

            new_pop.extend(sorted([t1, t2, o1, o2], key=lambda x: x.fitness)[-2:])

            if o1 in new_pop[-2:]:
                replaced += 1
            if o2 in new_pop[-2:]:
                replaced += 1

        else:
            new_pop.extend([t1, t2])

    population = new_pop
    #print(f"Replaced {replaced} organisms")

    for tree in population:
        if tree.fitness < optimal_fitness[-1]:
            optimal_fitness[-1] = tree.fitness
            optimal_tree = tree

    optimal_fitness.append(optimal_fitness[-1])

print(optimal_tree)
plot_predictions(optimal_tree, xs, ys)

plt.plot(range(N_GEN+1), optimal_fitness)
plt.show()