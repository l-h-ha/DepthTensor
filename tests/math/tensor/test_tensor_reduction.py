import numpy as np
import depthtensor as dt

def test_sum_forward_and_backward(device, assert_allclose):
    x_data = [[1.0, 2.0], [3.0, 4.0]]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    z = x.sum()

    expected_z = np.sum(x_data)
    assert z.shape == ()
    assert_allclose(z, expected_z)

    z.backward()

    expected_grad = np.ones_like(x_data)
    assert x.grad is not None
    assert_allclose(x.grad, expected_grad)

def test_mean_forward_and_backward(device, assert_allclose):
    x_data = [1.0, 2.0, 3.0, 4.0]
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    z = x.mean()

    assert_allclose(z, np.mean(x_data))

    z.backward()

    N = len(x_data)
    expected_grad = np.full_like(x_data, 1.0 / N)
    assert_allclose(x.grad, expected_grad)
