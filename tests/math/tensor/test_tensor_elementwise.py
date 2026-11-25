import numpy as np
import depthtensor as dt

def test_addition(device, assert_allclose):
    a_data, b_data = [1.0, 2.0], [3.0, 4.0]
    a = dt.Tensor(a_data, device=device, requires_grad=True)
    b = dt.Tensor(b_data, device=device, requires_grad=True)

    z = a + b
    assert_allclose(z, np.array(a_data) + np.array(b_data))

    loss = z.sum()
    loss.backward()

    assert_allclose(a.grad, np.ones_like(a_data))
    assert_allclose(b.grad, np.ones_like(b_data))

def test_multiplication(device, assert_allclose):
    a_data, b_data = [[1.0, 2.0]], [[3.0, 4.0]]
    a = dt.Tensor(a_data, device=device, requires_grad=True)
    b = dt.Tensor(b_data, device=device, requires_grad=True)

    z = a * b
    assert_allclose(z, np.array(a_data) * np.array(b_data))

    loss = z.sum()
    loss.backward()

    assert_allclose(a.grad, np.array(b_data))
    assert_allclose(b.grad, np.array(a_data))

def test_power(device, assert_allclose):
    a_data = [1.0, 2.0, 3.0]
    a = dt.Tensor(a_data, device=device, requires_grad=True)
    power = 3.0

    z = a**power
    assert_allclose(z, np.array(a_data) ** power)

    loss = z.sum()
    loss.backward()

    expected_grad = power * (np.array(a_data) ** (power - 1))
    assert_allclose(a.grad, expected_grad)

def test_exp_unary(device, assert_allclose):
    x_data = [0.0, 1.0, 2.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    z = x.exp()
    assert_allclose(z, np.exp(x_data))

    loss = z.sum()
    loss.backward()

    assert_allclose(x.grad, np.exp(x_data))

def test_matrix_multiplication(device, assert_allclose):
    a_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b_data = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

    a = dt.Tensor(a_data, device=device, requires_grad=True)
    b = dt.Tensor(b_data, device=device, requires_grad=True)

    z = a @ b

    expected_z = np.matmul(np.array(a_data), np.array(b_data))
    assert z.shape == (2, 2)
    assert_allclose(z, expected_z)

    loss = z.sum()
    loss.backward()

    grad_z = np.ones((2, 2))

    expected_grad_a = np.matmul(grad_z, np.transpose(b_data))
    expected_grad_b = np.matmul(np.transpose(a_data), grad_z)

    assert_allclose(a.grad, expected_grad_a)
    assert_allclose(b.grad, expected_grad_b)

def test_division(device, assert_allclose):
    x_data, y_data = [2.0, 4.0], [2.0, 2.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)
    y = dt.Tensor(y_data, device=device, requires_grad=True)

    z = x / y
    assert_allclose(z, np.array(x_data) / np.array(y_data))

    loss = z.sum()
    loss.backward()

    x_np, y_np = np.array(x_data), np.array(y_data)
    expected_grad_x = 1.0 / y_np
    expected_grad_y = -x_np / (y_np**2)

    assert_allclose(x.grad, expected_grad_x)
    assert_allclose(y.grad, expected_grad_y)

def test_log(device, assert_allclose):
    x_data = [1.0, 2.0, 10.0]  # Must be strictly > 0 for log
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    z = x.log()
    assert_allclose(z, np.log(x_data))

    loss = z.sum()
    loss.backward()

    expected_grad = 1.0 / np.array(x_data)
    assert_allclose(x.grad, expected_grad)

def test_sqrt(device, assert_allclose):
    x_data = [1.0, 4.0, 9.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    z = x.sqrt()
    assert_allclose(z, np.sqrt(x_data))

    loss = z.sum()
    loss.backward()

    expected_grad = 1.0 / (2.0 * np.sqrt(x_data))
    assert_allclose(x.grad, expected_grad)

def test_square(device, assert_allclose):
    x_data = [1.0, 2.0, 3.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    z = x.square()
    assert_allclose(z, np.square(x_data))

    loss = z.sum()
    loss.backward()

    expected_grad = 2.0 * np.array(x_data)
    assert_allclose(x.grad, expected_grad)

def test_clip(device, assert_allclose):
    x_data = [0.0, 2.0, 5.0, 10.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    min_val, max_val = 2.0, 5.0

    z = x.clip(min_val, max_val)
    assert_allclose(z, np.clip(x_data, min_val, max_val))

    loss = z.sum()
    loss.backward()

    x_np = np.array(x_data)
    expected_grad = np.where((x_np >= min_val) & (x_np <= max_val), 1.0, 0.0)
    assert_allclose(x.grad, expected_grad)
