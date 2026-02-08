import pytest
import numpy as np
from src.depthtensor import Tensor, differentiate
import src.depthtensor._core.ops.elementwise as el
import src.depthtensor as dt

def test_autodiff_add():
    x = Tensor([1.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = x + y
    differentiate(z)
    
    assert np.allclose(x.grad, [1.0])
    assert np.allclose(y.grad, [1.0])

def test_autodiff_mul():
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    z = x * y
    differentiate(z)
    
    assert np.allclose(x.grad, [3.0]) # dz/dx = y
    assert np.allclose(y.grad, [2.0]) # dz/dy = x

def test_autodiff_pow():
    x = Tensor([2.0], requires_grad=True)
    z = x ** 3
    differentiate(z)
    
    assert np.allclose(x.grad, [3 * (2.0**2)]) # 3x^2

def test_autodiff_chain_rule():
    # z = (x + y) * x
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    a = x + y
    z = a * x
    differentiate(z)
    
    # dz/dx = d(x^2 + xy)/dx = 2x + y = 4 + 3 = 7
    # dz/dy = d(x^2 + xy)/dy = x = 2
    
    assert np.allclose(x.grad, [7.0])
    assert np.allclose(y.grad, [2.0])

def test_autodiff_broadcasting():
    # x: (1,), y: (3,)
    # z = x + y
    x = Tensor([1.0], requires_grad=True)
    y = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = x + y
    # z is [2, 3, 4]
    # Sum of z is scalar to differentiate, but default differentiate assumes sum if tensor is not scalar? 
    # Actually differentiate sets grad to ones if None.
    # If z is vector, grad is ones(3).
    # x contributes to all 3 elements.
    differentiate(z)
    
    # dz_i/dx = 1. Sum over i = 3.
    assert np.allclose(x.grad, [3.0])
    # dz_i/dy_j = delta_ij. Sum over i => 1.
    assert np.allclose(y.grad, [1.0, 1.0, 1.0])

def test_autodiff_matmul():
    # A (1x2) @ B (2x1) = C (1x1)
    A = Tensor([[1.0, 2.0]], requires_grad=True)
    B = Tensor([[3.0], [4.0]], requires_grad=True)
    C = A @ B
    differentiate(C)
    
    # C = 1*3 + 2*4 = 11
    # dC/dA = B.T = [3, 4]
    # dC/dB = A.T = [[1], [2]]
    
    assert np.allclose(A.grad, [[3.0, 4.0]])
    assert np.allclose(B.grad, [[1.0], [2.0]])

def test_autodiff_indexing():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x[0] * 2
    differentiate(y)
    
    # y = 2 * x[0]
    # dy/dx[0] = 2, dy/dx[1] = 0, dy/dx[2] = 0
    assert np.allclose(x.grad, [2.0, 0.0, 0.0])

def test_rosenbrock_step():
    # Test one step of optimization logic
    x = Tensor([1.2], requires_grad=True)
    y = Tensor([1.2], requires_grad=True)
    a, b = Tensor([1.0]), Tensor([100.0])
    
    # f = (a-x)^2 + b(y-x^2)^2
    loss = (a - x) ** 2 + b * (y - x**2) ** 2
    differentiate(loss)
    
    # Analytical gradients
    # df/dx = -2(a-x) + 2b(y-x^2)(-2x)
    #       = -2(1-1.2) - 400(1.2 - 1.44)(1.2)
    #       = -2(-0.2) - 400(-0.24)(1.2)
    #       = 0.4 + 115.2 = 115.6
    
    # df/dy = 2b(y-x^2)
    #       = 200(1.2 - 1.44)
    #       = 200(-0.24)
    #       = -48.0
    
    assert np.isclose(x.grad.item(), 115.6)
    assert np.isclose(y.grad.item(), -48.0)

def test_neg_autodiff():
    x = Tensor([1.0], requires_grad=True)
    y = -x
    differentiate(y)
    assert np.allclose(x.grad, [-1.0])

def test_mean_autodiff():
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = x.mean()
    differentiate(y)
    # y = 1/4 sum(x)
    # dy/dx = 1/4
    assert np.allclose(x.grad, [0.25, 0.25, 0.25, 0.25])

def test_reuse_graph():
    x = Tensor([2.0], requires_grad=True)
    y = x * x
    differentiate(y)
    assert np.allclose(x.grad, [4.0])
    
    x.zero_grad()
    # Need to recompute forward to get new graph connections if we were dynamic?
    # This library builds graph dynamically on forward.
    z = x * x * x
    differentiate(z)
    assert np.allclose(x.grad, [12.0]) # 3x^2 = 12

def test_accumulate_grad():
    x = Tensor([2.0], requires_grad=True)
    y = x * x
    differentiate(y)
    assert np.allclose(x.grad, [4.0])
    
    # Do NOT zero grad
    z = x * x # Another x^2
    differentiate(z)
    # Grad should accumulate: 4.0 + 4.0 = 8.0
    assert np.allclose(x.grad, [8.0])

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

def test_autodiff_sum():
    x = Tensor([1.0, 2.0], requires_grad=True)
    z = x.sum(requires_grad=True)
    differentiate(z)
    
    # z = x[0] + x[1]
    # dz/dx = [1.0, 1.0]
    assert np.allclose(x.grad, [1.0, 1.0])

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
