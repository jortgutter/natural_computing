import matplotlib.pyplot as plt
import numpy as np

from tree import Tree
from tqdm import tqdm
from node import *
import copy


def load_data(filename: str) -> Tuple[List[float], List[float]]:
    """
    This function loads the target data from the specified file
    :param filename: The file that contains the target data
    :return: xs: x-values, ys: y-values
    """
    with open(filename, "r") as file:
        xs = [float(term) for term in file.readline().split(", ")]
        ys = [float(term) for term in file.readline().split(", ")]
    return xs, ys


def get_predictions(tree: Tree, terminal_value: TerminalValue, x_vals: List[float]) -> List[float]:
    """
    This uses the given tree to predict the target values
    :param tree: The tree to predict with
    :param terminal_value: The leaf node object to put the x-values in
    :param x_vals: the x-values
    :return: the predicted y-values
    """
    results = []

    for x in x_vals:
        terminal_value.v = x
        results.append(tree.root.evaluate())

    return results


def plot_predictions(tree: Tree, xs: List[float], ys: List[float]):
    """
    This function plots the predictions of a tree versus the target data
    :param tree: The tree to predict with
    :param xs: x-values
    :param ys: target y-values
    """
    res = get_predictions(tree, term_val, xs)

    plt.plot(xs, ys, label="True", c="green")
    plt.plot(xs, res, label="Estimated", c="black")
    plt.xlabel("$x$")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()


def plot_fitness(fitness: List[float]):
    """
    This function plots the max fitness per iteration
    :param fitness: The max fitness per iteration
    """
    plt.plot(fitness)
    plt.xlabel("Iteration")
    plt.ylabel("Max. fitness")
    plt.title("Maximum fitness per iteration")
    plt.show()


def plot_sizes(sizes: List[int]):
    """
    This function plots the size of the best organism per iteration
    :param sizes: The sizes of the best organisms
    """
    plt.plot(sizes)
    plt.xlabel("Iteration")
    plt.ylabel("Number of nodes")
    plt.title("Number of nodes of best organism per iteration")
    plt.show()


def crossover(tree1: Tree, tree2: Tree):
    """
    This function performs crossover between tree1 and tree2.
    Nothing is returned, as the trees are manipulated directly
    :param tree1: First tree to perform crossover on
    :param tree2: Second tree to perform crossover on
    """
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


def protected_div(a: float, b: float) -> float:
    return 1 if b == 0 else np.divide(a, b)


def protected_log(a: float) -> float:
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

MAX_DEPTH = 8
POP_SIZE = 150

N_GEN = 50
P_CROSSOVER = 0.7
k = int(.3*POP_SIZE)


def calculate_fitness(tree: Tree, term_val: TerminalValue, xs: List[float], ys: List[float]) -> float:
    """
    This function calculates the fitness of a tree given the target y-values
    :param tree: The tree to evaluate
    :param term_val: The leaf node object
    :param xs: the x-values
    :param ys: the target y-values
    :return: The negative sum of absolute errors
    """
    preds = get_predictions(tree, term_val, xs)

    errors = [abs(y - pred) for y, pred in zip(ys, preds)]

    return - sum(errors)


def select_parents(population: List[Tree], k: int) -> Tuple[Tree, Tree]:
    """
    This function selects two parents from the specified population by means of two tournaments of group size k
    :param population: The population to select parents from
    :param k: The tournament group size
    :return: The two selected parents
    """
    group1 = np.random.permutation(population)[:k]
    group2 = np.random.permutation(population)[:k]

    return get_best(group1), get_best(group2)


def get_best(group: np.ndarray) -> Tree:
    """
    This function selects the best tree from a group of trees, based on the (pre-calculated) fitness
    :param group: The group to select the best tree from
    :return: The best tree
    """
    best_fit = -np.inf
    best_tree = None

    for tree in group:
        if tree.fitness > best_fit:
            best_fit = tree.fitness
            best_tree = tree

    return best_tree


population = [Tree(FunctionNode(None, functions=functions, terminal_value=term_val, max_depth=MAX_DEPTH)) for _ in range(POP_SIZE)]


optimal_fitness = [-np.inf]
optimal_size = [-1]
optimal_tree = None
for tree in population:
    fitness = calculate_fitness(tree, term_val, xs, ys)
    if fitness > optimal_fitness[0]:
        optimal_fitness[0] = fitness
        optimal_size[0] = tree.get_node_count()
        optimal_tree = tree
    tree.update_fitness(fitness)

for gen in tqdm(range(N_GEN)):  # For each generation
    replaced = 0
    new_pop = []

    for i in range(0, POP_SIZE, 2):  # Generate offspring POP_SIZE/2 times and select the best 2 from parents + offspring
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

    for tree in population:  # Update statistics of best organism
        if tree.fitness > optimal_fitness[-1]:
            optimal_fitness[-1] = tree.fitness
            optimal_size[-1] = tree.get_node_count()
            optimal_tree = tree

    optimal_fitness.append(optimal_fitness[-1])
    optimal_size.append(optimal_size[-1])

# Display statistics
print(optimal_tree)
plot_predictions(optimal_tree, xs, ys)
plot_fitness(optimal_fitness)
plot_sizes(optimal_size)
