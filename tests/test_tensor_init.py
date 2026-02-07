import pytest
import numpy as np
from src.depthtensor import Tensor
import src.depthtensor as dt

def test_tensor_init_list():
    t = Tensor([1, 2, 3])
    assert isinstance(t.data, np.ndarray)
    assert np.array_equal(t.data, np.array([1, 2, 3]))
    assert t.device == "cpu"
    assert t.requires_grad is False

def test_tensor_init_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = Tensor(arr, requires_grad=True)
    assert np.array_equal(t.data, arr)
    assert t.requires_grad is True
    assert t.shape == (2, 2)
    assert t.dtype == np.float64

def test_tensor_properties():
    t = Tensor(np.zeros((2, 3, 4)))
    assert t.shape == (2, 3, 4)
    assert t.ndim == 3
    assert t.size == 24
    
def test_tensor_item():
    t = Tensor([42])
    assert t.item() == 42
    
def test_tensor_copy():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = t1.copy()
    assert np.array_equal(t1.data, t2.data)
    assert t1 is not t2
    assert t2.requires_grad is False # Default is False for copy unless specified

    t3 = t1.copy(copy_requires_grad=True)
    assert t3.requires_grad is True

def test_make_differentiable():
    t = Tensor([1.0])
    assert not t.requires_grad
    t.make_differentiable()
    assert t.requires_grad
    assert t.grad is not None
    assert np.all(t.grad == 0)

def test_zero_grad():
    t = Tensor([1.0], requires_grad=True)
    t.make_differentiable()
    t.grad = np.array([0.5])
    t.zero_grad()
    assert np.all(t.grad == 0)

def test_zero_grad_error():
    t = Tensor([1.0], requires_grad=False)
    with pytest.raises(RuntimeError):
        t.zero_grad()
