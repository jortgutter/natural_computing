from dataclasses import dataclass
from node import *


@dataclass
class Tree:
    """
    Simple wrapper to easily keep track of a tree, even if the root node changes
    """
    root: Node
    fitness: float = -1

    def update_root(self, new_root):
        """
        Update the root node of the tree
        :param new_root: The new root node
        """
        self.root = new_root

    def update_fitness(self, fitness):
        """
        Set the fitness of the tree
        :param fitness: The (pre-calculated) fitness of the tree
        """
        self.fitness = fitness

    def get_node_count(self) -> int:
        """
        Get the number of nodes in the tree
        :return: Number of nodes
        """
        return self.root.get_node_count()

    def __repr__(self):
        return self.root.__repr__()
