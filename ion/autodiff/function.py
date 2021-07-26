"""
Contains the definition of the `ion.autodiff.Function` interface.
"""
from abc import ABCMeta, abstractmethod

from ion.datatypes import Tensor


class Function(object, metaclass=ABCMeta):
    @abstractmethod
    def backward(self, gradient: Tensor) -> None:
        pass

    @abstractmethod
    def forward(self) -> "Variable":
        pass

    def __call__(self) -> "Variable":
        return self.forward()
