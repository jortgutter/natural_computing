from dataclasses import dataclass
from typing import *
import numpy as np

newline = "\n"
tab = "\t"


@dataclass
class TerminalValue:
    v: float


class Node:
    """
    Base class of a node. Could be either a Function or Terminal.
    """
    def __init__(self, parent):
        self.p = None
        self.children = False
        self.depth = 0
        self.update_parent(parent)

    def update_parent(self, new_parent):
        """
        Update parent of the current node. Also update the depth attribute
        :param new_parent: The new parent of the current node
        """
        self.p = new_parent
        self.update_depth()

    def update_depth(self):
        """
        Update depth of this node depending on parent depth
        """
        self.depth = 0 if not self.p else self.p.get_depth() + 1

    def get_depth(self) -> int:
        """
        Get the depth of the current node
        :return: Depth of the current node
        """
        return self.depth

    def evaluate(self) -> float:
        pass

    def get_node_count(self) -> int:
        pass

    def get_reference_list(self) -> list:
        """
        Return the reference to the current node in a list
        :return: Reference of current node
        """
        return [self]


class FunctionNode(Node):
    """
    A Node representing a unary or binary operator.
    The function of the Node is randomly selected from a list of functions.
    """
    def __init__(self, parent, functions: List[Tuple[Any, int]], terminal_value: Optional[TerminalValue] = None, max_depth: int = 0):
        super().__init__(parent)

        f_idx = np.random.choice(range(len(functions)))
        fun = functions[f_idx]
        self.f, self.arity = fun

        self.max_depth = max_depth
        self.children = [(FunctionNode(self, functions, terminal_value=terminal_value, max_depth=max_depth)
                          if np.random.uniform(0, 1) < 0.75 else
                          TerminalNode(self, value=terminal_value)
                          )if self.depth < self.max_depth-1 else
                         TerminalNode(self, value=terminal_value) for _ in range(self.arity)]

    def evaluate(self) -> float:
        """
        Get the value of this subtree by evaluating the children and passing their values through the function
        :return: The resulting value
        """
        return self.f(*[c.evaluate() for c in self.children])

    def get_node_count(self) -> int:
        """
        Count the current node and recursively its children
        :return: Number of nodes in the subtree
        """
        return 1 + sum([c.get_node_count() for c in self.children])

    def update_depth(self):
        """
        Update the depth of the current node and recursively also of its children
        """
        super().update_depth()

        if self.children:
            for c in self.children:
                c.update_depth()

    def get_reference_list(self) -> list:
        """
        Return a list of references of the current node and its children
        :return:
        """
        ls = super().get_reference_list()
        for c in self.children:
            ls.extend(c.get_reference_list())
        return ls

    def __repr__(self) -> str:
        return "\t"*self.depth + f"- Function[{self.depth}]: {self.f}\n{newline.join([str(c) for c in self.children])}"


class TerminalNode(Node):
    """
    A Node representing a terminal.
    The value, if not specified, is randomly drawn from a distribution Uniform(0, 1).
    """
    def __init__(self, parent, value: Optional[TerminalValue] = None):
        super().__init__(parent)

        self.value = value if value else TerminalValue(np.random.uniform(0, 1))

    def evaluate(self) -> float:
        """
        Evaluate this terminal node
        :return: Value of the terminal node
        """
        return self.value.v

    def get_node_count(self) -> int:
        """
        Return 1 for this terminal node
        :return: 1
        """
        return 1

    def __repr__(self) -> str:
        return "\t"*self.depth + f"- Terminal: {self.value.v}"
