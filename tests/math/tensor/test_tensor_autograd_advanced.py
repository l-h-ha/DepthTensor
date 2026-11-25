import pytest
import numpy as np
import depthtensor as dt

def test_slice_backward(device, assert_allclose):
    x_data = [1.0, 2.0, 3.0, 4.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    y = x[1:3]
    assert_allclose(y, [2.0, 3.0])

    loss = y.sum()
    loss.backward()

    expected_grad = [0.0, 1.0, 1.0, 0.0]
    assert_allclose(x.grad, expected_grad)

def test_axis_reduction_backward(device, assert_allclose):
    x_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    y = x.sum(axis=0)
    assert_allclose(y, [5.0, 7.0, 9.0])

    loss = y.sum()
    loss.backward()

    expected_grad = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert_allclose(x.grad, expected_grad)

def test_batched_matmul(device, assert_allclose):
    a_data = np.random.randn(2, 2, 3)
    b_data = np.random.randn(2, 3, 4)

    a = dt.Tensor(a_data, device=device, requires_grad=True)
    b = dt.Tensor(b_data, device=device, requires_grad=True)

    z = a @ b
    expected_z = np.matmul(a_data, b_data)

    assert z.shape == (2, 2, 4)
    assert_allclose(z, expected_z)

    loss = z.sum()
    loss.backward()

    grad_z = np.ones((2, 2, 4))

    expected_grad_a = np.matmul(grad_z, np.swapaxes(b_data, -1, -2))
    expected_grad_b = np.matmul(np.swapaxes(a_data, -1, -2), grad_z)

    assert_allclose(a.grad, expected_grad_a)
    assert_allclose(b.grad, expected_grad_b)

def test_abs_and_sign_backward(device, assert_allclose):
    x_data = [-3.0, 0.0, 2.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    z_abs = x.abs()
    assert_allclose(z_abs, [3.0, 0.0, 2.0])

    z_abs.sum().backward()
    assert_allclose(x.grad, [-1.0, 0.0, 1.0])

    x.zero_grad()

    z_sign = x.sign()
    assert_allclose(z_sign, [-1.0, 0.0, 1.0])

    z_sign.sum().backward()
    assert_allclose(x.grad, [0.0, 0.0, 0.0])

def test_maximum_backward_selects_larger_operand(device, assert_allclose):
    a = dt.Tensor([1.0, 5.0, 3.0], device=device, requires_grad=True)
    b = dt.Tensor([2.0, 4.0, 4.0], device=device, requires_grad=True)

    z = a.maximum(b)
    assert_allclose(z, [2.0, 5.0, 4.0])

    loss = z.sum()
    loss.backward()

    assert_allclose(a.grad, [0.0, 1.0, 0.0])
    assert_allclose(b.grad, [1.0, 0.0, 1.0])

@pytest.mark.parametrize("use_tensor_seed", [True, False])
def test_custom_gradient_seeding(device, assert_allclose, use_tensor_seed):
    x_data = [2.0, 3.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)
    z = x * 2.0

    incoming_grad = dt.Tensor([10.0, 20.0], device=device)
    seed = incoming_grad if use_tensor_seed else incoming_grad.data
    z.backward(grad=seed)

    assert_allclose(x.grad, [20.0, 40.0])

    z.backward()
    assert_allclose(x.grad, [22.0, 42.0])
    assert_allclose(incoming_grad, [10.0, 20.0])

def test_add_rejects_mixed_devices():
    import importlib.util

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is unavailable.")

    a = dt.Tensor([1.0], device="cpu", requires_grad=True)
    b = dt.Tensor([2.0], device="cuda", requires_grad=True)

    with pytest.raises(Exception):
        a + b
