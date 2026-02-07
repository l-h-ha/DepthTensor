import pytest
import numpy as np
from src.depthtensor import Tensor, differentiate
import src.depthtensor._core.ops.elementwise as el

def test_autodiff_sub():
    x = Tensor([5.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = x - y
    differentiate(z)
    assert np.allclose(x.grad, [1.0])
    assert np.allclose(y.grad, [-1.0])

def test_autodiff_div():
    x = Tensor([10.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = x / y
    differentiate(z)
    # z = x * y^-1
    # dz/dx = 1/y = 0.5
    # dz/dy = -x/y^2 = -10/4 = -2.5
    assert np.allclose(x.grad, [0.5])
    assert np.allclose(y.grad, [-2.5])

def test_autodiff_exp():
    x = Tensor([2.0], requires_grad=True)
    z = el.exp(x, differentiate=True)
    differentiate(z)
    # z = e^x
    # dz/dx = e^x
    assert np.allclose(x.grad, np.exp([2.0]))

def test_autodiff_log():
    x = Tensor([2.0], requires_grad=True)
    z = el.log(x, differentiate=True)
    differentiate(z)
    # z = ln(x)
    # dz/dx = 1/x = 0.5
    assert np.allclose(x.grad, [0.5])

def test_autodiff_sqrt():
    x = Tensor([4.0], requires_grad=True)
    z = el.sqrt(x, differentiate=True)
    differentiate(z)
    # z = sqrt(x)
    # dz/dx = 1/(2sqrt(x)) = 1/4 = 0.25
    assert np.allclose(x.grad, [0.25])

def test_autodiff_square():
    x = Tensor([3.0], requires_grad=True)
    z = el.square(x, differentiate=True)
    differentiate(z)
    # z = x^2
    # dz/dx = 2x = 6
    assert np.allclose(x.grad, [6.0])

def test_autodiff_abs():
    x = Tensor([-2.0], requires_grad=True)
    z = el.abs(x, differentiate=True)
    differentiate(z)
    # z = |x|
    # dz/dx = sign(x) = -1
    assert np.allclose(x.grad, [-1.0])

def test_autodiff_transpose():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    z = x.transpose(None) # Transpose
    differentiate(z)
    # dz/dx_ij should correspond to mapping.
    # z_ji = x_ij
    # if we sum(z) implicitly (grad=ones), then d(sum z)/dx_ij = 1.
    assert np.allclose(x.grad, [[1.0, 1.0], [1.0, 1.0]])

def test_autodiff_sum_not_linked():
    # Confirming sum is not currently differentiable in the graph sense
    # It returns a Tensor with requires_grad=True but no backward link in this version
    x = Tensor([1.0, 2.0], requires_grad=True)
    z = x.sum(requires_grad=True)
    
    # differentiate(z) should probably fail or do nothing if no backward is attached
    # implementation: if t.backward is None and prev is empty, it continues.
    # sum returns a tensor with NO prev (it's created fresh).
    # So differentiate stops immediately.
    differentiate(z)
    
    assert x.grad is None or np.all(x.grad == 0) # Should be None actually if not initialized

def test_max_not_linked():
    x = Tensor([1.0, 5.0], requires_grad=True)
    z = x.max(requires_grad=True)
    differentiate(z)
    assert x.grad is None or np.all(x.grad == 0)

def test_maximum_not_linked():
    x = Tensor([1.0, 5.0], requires_grad=True)
    y = Tensor([2.0, 2.0], requires_grad=True)
    z = x.maximum(y, requires_grad=True)
    differentiate(z)
    assert x.grad is None or np.all(x.grad == 0)
    assert y.grad is None or np.all(y.grad == 0)

def test_clip_not_linked():
    x = Tensor([1.0, 10.0], requires_grad=True)
    z = x.clip(2, 5, requires_grad=True)
    differentiate(z)
    assert x.grad is None or np.all(x.grad == 0)

def test_autodiff_power_tensor():
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    z = x ** y
    differentiate(z)
    # z = x^y
    # dz/dx = y * x^(y-1) = 3 * 2^2 = 12
    # dz/dy = x^y * ln(x) = 8 * ln(2)
    assert np.allclose(x.grad, [12.0])
    assert np.allclose(y.grad, [8.0 * np.log(2.0)])

def test_autodiff_mean_axis():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # Mean over axis 0 -> [2.0, 3.0]
    z = x.mean(axis=0) 
    differentiate(z)
    # z_j = 0.5 * (x_0j + x_1j)
    # dz_j / dx_ij = 0.5
    assert np.allclose(x.grad, [[0.5, 0.5], [0.5, 0.5]])
