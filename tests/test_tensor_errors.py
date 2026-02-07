import pytest
import numpy as np
from src.depthtensor import Tensor

def test_inplace_add_error():
    x = Tensor([1.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations"):
        x += 1

def test_inplace_sub_error():
    x = Tensor([1.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations"):
        x -= 1

def test_inplace_mul_error():
    x = Tensor([1.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations"):
        x *= 2

def test_inplace_div_error():
    x = Tensor([1.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations"):
        x /= 2

def test_setitem_error():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations"):
        x[0] = 5.0

def test_to_device_inplace_error():
    x = Tensor([1.0], requires_grad=True)
    # Must switch device to trigger the check
    # We expect RuntimeError because of in-place on differentiable tensor
    # This check happens BEFORE actual data conversion/cupy check
    with pytest.raises(RuntimeError, match="In-place operations"):
        x.to_device("gpu", in_place=True)

def test_zero_grad_undifferentiable_error():
    x = Tensor([1.0], requires_grad=False)
    with pytest.raises(RuntimeError):
        x.zero_grad()

def test_make_differentiable_grad_type_error():
    x = Tensor([1.0])
    with pytest.raises(RuntimeError):
        # Passing list instead of ndarray/Tensor
        x.make_differentiable(grad=[1.0])
