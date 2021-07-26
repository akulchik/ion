"""
Contains the implementation of `ion.autodiff.Variable`.
"""
from typing import Tuple

import numpy as np

from ion.datatypes import Tensor


class Variable(object):
    """
    The `Variable` class is a wrapper class that is used to represent data by
    the automatic differentiation `autodiff` module. It comes in handy to
    represent not only data fed to a neural network, but also all intermediate
    calculations results, and most importantly their gradients.
    """
    def __init__(self, data: Tensor, requires_gradient: bool = False,
                 parent_func: "Function" = None) -> None:
        self.data = np.array(data)
        self.requires_gradient = requires_gradient
        self.gradient = None
        self.parent_func = parent_func

    def backward(self, gradient: Tensor) -> None:
        """
        Compute the gradient of the current variable w.r.t. the subgraph
        sources.

        :param gradient: A `Tensor` giving the gradient w.r.t. the current
            variable.
        :return: `None`.
        """
        if self.requires_gradient:
            self.gradient = gradient
        if not self.is_source:
            self.parent_func.backward(gradient)

    @property
    def is_source(self) -> bool:
        """
        Check if the node is a source node. A source variable node is a node
        that is allowed to have only outgoing connections to function nodes,
        but not to other variable nodes.

        :return: `True` iff the node is a source.
        """
        return self.parent_func is None

    @property
    def shape(self) -> Tuple:
        """
        Return the shape of data wrapped by the variable.

        :return: A `tuple` giving the shape of data.
        """
        return self.data.shape
