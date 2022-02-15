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
        self.p = new_parent
        self.update_depth()

    def update_depth(self):
        self.depth = 0 if not self.p else self.p.get_depth() + 1

    def get_depth(self) -> int:
        return self.depth

    def evaluate(self) -> float:
        pass

    def get_node_count(self) -> int:
        pass

    def get_reference_list(self) -> list:
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
                          if np.random.uniform(0, 1) < 0.5 else
                          TerminalNode(self, value=terminal_value)
                          )if self.depth < self.max_depth-1 else
                         TerminalNode(self, value=terminal_value) for _ in range(self.arity)]

    def evaluate(self) -> float:
        return self.f(*[c.evaluate() for c in self.children])

    def get_node_count(self) -> int:
        return 1 + sum([c.get_node_count() for c in self.children])

    def update_depth(self):
        super().update_depth()

        if self.children:
            for c in self.children:
                c.update_depth()

    def get_reference_list(self) -> list:
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
        return self.value.v

    def get_node_count(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "\t"*self.depth + f"- Terminal: {self.value.v}"
