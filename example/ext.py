import argparse

import numpy as np

import depthtensor as dt

def _to_numpy(array):
    if hasattr(array, "get"):
        return array.get()

    return np.asarray(array)

def test_tensor_multiplication(device):
    x_data = [[1.0, 2.0], [3.0, 4.0]]
    y_data = [[2.0, 0.5], [1.0, 2.0]]

    x = dt.Tensor(x_data, device=device, requires_grad=True)
    y = dt.Tensor(y_data, device=device, requires_grad=True)

    z = x * y

    expected_z = np.array(x_data) * np.array(y_data)
    np.testing.assert_allclose(_to_numpy(z.data), expected_z)

    loss = z.sum()
    loss.backward()

    np.testing.assert_allclose(_to_numpy(x.grad), y_data)
    np.testing.assert_allclose(_to_numpy(y.grad), x_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check multiplication and gradients.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    test_tensor_multiplication(parser.parse_args().device)
