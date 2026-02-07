import pytest
import numpy as np
from src.depthtensor import Tensor
import src.depthtensor as dt

def test_add():
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 + t2
    assert np.array_equal(t3.data, np.array([4, 6]))

def test_sub():
    t1 = Tensor([5, 6])
    t2 = Tensor([1, 2])
    t3 = t1 - t2
    assert np.array_equal(t3.data, np.array([4, 4]))

def test_mul():
    t1 = Tensor([2, 3])
    t2 = Tensor([4, 5])
    t3 = t1 * t2
    assert np.array_equal(t3.data, np.array([8, 15]))

def test_div():
    t1 = Tensor([10.0, 20.0])
    t2 = Tensor([2.0, 4.0])
    t3 = t1 / t2
    assert np.array_equal(t3.data, np.array([5.0, 5.0]))

def test_pow():
    t1 = Tensor([2.0, 3.0])
    t3 = t1 ** 2
    assert np.array_equal(t3.data, np.array([4.0, 9.0]))

def test_matmul():
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[1, 0], [0, 1]])
    t3 = t1 @ t2
    assert np.array_equal(t3.data, t1.data)

def test_neg():
    t1 = Tensor([1, -2])
    t2 = -t1
    assert np.array_equal(t2.data, np.array([-1, 2]))

def test_comparison():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([1, 0, 4])
    
    assert np.array_equal((t1 == t2).data, np.array([True, False, False]))
    assert np.array_equal((t1 > t2).data, np.array([False, True, False]))
    assert np.array_equal((t1 < t2).data, np.array([False, False, True]))

def test_sum():
    t = Tensor([[1, 2], [3, 4]])
    assert t.sum().item() == 10
    assert np.array_equal(t.sum(axis=0, keepdims=False).data, np.array([4, 6]))

def test_sum_initial():
    t = Tensor([1, 2, 3])
    # sum = 6. initial = 10. result should be 16.
    res = t.sum(initial=10)
    assert res.item() == 16

def test_mean():
    t = Tensor([1.0, 2.0, 3.0, 4.0])
    assert t.mean().item() == 2.5

def test_max():
    t = Tensor([1, 5, 2])
    assert t.max().item() == 5

def test_clip():
    t = Tensor([1, 10, 5])
    clipped = t.clip(2, 6)
    assert np.array_equal(clipped.data, np.array([2, 6, 5]))

def test_indexing():
    t = Tensor([10, 20, 30])
    assert t[1].item() == 20
    t[1] = 25
    assert t[1].item() == 25
