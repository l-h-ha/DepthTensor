from DepthTensor import *

import numpy as np
try:
    import cupy as cp
except:
    cp = None
import pytest

###
###
###

@pytest.fixture
def tensor_1_1():
    return CPUTensor(1, dtype=float64, requires_grad=True)

@pytest.fixture
def tensor_1_2():
    return CPUTensor(1., dtype=float16, requires_grad=True)

@pytest.fixture
def tensor_1_3():
    return CPUTensor([1., 2., 3.], dtype=float16, requires_grad=True)

###
###
###

def test_attr(
        tensor_1_1: CPUTensor,
        tensor_1_2: CPUTensor,
        tensor_1_3: CPUTensor
    ):

    assert isinstance(tensor_1_1.data, np.ndarray)
    assert tensor_1_1.dtype == float64
    assert tensor_1_1.data.dtype == float64
    assert tensor_1_1.requires_grad == True

    assert isinstance(tensor_1_2.data, np.ndarray)
    assert tensor_1_2.dtype == float16
    assert tensor_1_2.data.dtype == float16
    assert tensor_1_2.requires_grad == True

    assert isinstance(tensor_1_3.data, np.ndarray)
    assert tensor_1_3.dtype == float16
    assert tensor_1_3.data.dtype == float16
    assert tensor_1_3.requires_grad == True