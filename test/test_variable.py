"""
Contains unit-tests covering the implementation of `ion.Variable`.
"""
from unittest.mock import Mock

import numpy as np
import pytest

from ion.autodiff import Function, Variable
from ion.datatypes import Tensor


@pytest.fixture
def data_and_gradient():
    shape = (4, 6)
    return np.zeros(shape), np.random.randn(*shape)


@pytest.fixture
def mock_op():
    return Mock(spec=Function)


def test_variable_default_init(data_and_gradient):
    data, _ = data_and_gradient
    var = Variable(data)
    assert isinstance(var.data, Tensor), "Data stored in a `Variable` is not a" \
                                         " `Tensor`"
    assert var.is_source, "`Variable` is not source by default, but it should be"
    assert not var.requires_gradient, "`Variable` requires gradient by default," \
                                      " but it shouldn't"
    assert var.gradient is None, "`Variable` gradient is not `None` when " \
                                 "default initialized, but it should be"


def test_variable_backward_given_is_source_and_requires_gradient(data_and_gradient):
    data, gradient = data_and_gradient
    var = Variable(data, requires_gradient=True)
    assert var.gradient is None
    var.backward(gradient)
    assert np.all(var.gradient == gradient), "`Variable` gradient is not saved during " \
                                             "`backward` if `requires_gradient` is True"


def test_variable_backward_given_is_not_source_and_requires_gradient(data_and_gradient, mock_op):
    data, gradient = data_and_gradient
    var = Variable(data, requires_gradient=True)
    var.parent_func = mock_op
    assert var.gradient is None
    assert not var.is_source
    var.backward(gradient)
    assert np.all(var.gradient == gradient)
    var.parent_func.backward.assert_called()


def test_variable_backward_given_is_source_and_not_requires_gradient(data_and_gradient):
    data, gradient = data_and_gradient
    var = Variable(data, requires_gradient=False)
    assert var.gradient is None
    var.backward(gradient)
    assert var.gradient is None


def test_variable_backward_given_is_not_source_and_not_requires_gradient(data_and_gradient, mock_op):
    data, gradient = data_and_gradient
    var = Variable(data, requires_gradient=False)
    var.parent_func = mock_op
    assert var.gradient is None
    assert not var.is_source
    var.backward(gradient)
    assert var.gradient is None
    var.parent_func.backward.assert_called()


def test_variable_shape(data_and_gradient):
    data, _ = data_and_gradient
    var = Variable(data)
    assert var.shape == data.shape
