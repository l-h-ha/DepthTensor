import pytest
import operator
import numpy as np
import depthtensor as dt

@pytest.mark.parametrize(
    "op_func, expected_grad_x, expected_grad_y",
    [
        (
            operator.add,
            [[2.0], [2.0]],  # Summing [[1, 1], [1, 1]] across axis 1
            [[2.0, 2.0]],  # Summing [[1, 1], [1, 1]] across axis 0
        ),
        (
            operator.sub,
            [[2.0], [2.0]],  # Summing [[1, 1], [1, 1]] across axis 1
            [[-2.0, -2.0]],  # Summing [[-1, -1], [-1, -1]] across axis 0
        ),
        (
            operator.mul,
            [[9.0], [9.0]],  # Summing y=[[4, 5], [4, 5]] across axis 1
            [[5.0, 5.0]],  # Summing x=[[2, 2], [3, 3]] across axis 0
        ),
        (
            operator.truediv,
            [[0.45], [0.45]],  # Summing 1/y across axis 1
            [[-0.3125, -0.2]],  # Summing -x/y^2 across axis 0
        ),
    ],
)
def test_binary_broadcasting(
    device, assert_allclose, op_func, expected_grad_x, expected_grad_y
):
    x_data = [[2.0], [3.0]]
    y_data = [[4.0, 5.0]]

    x = dt.Tensor(x_data, device=device, requires_grad=True)
    y = dt.Tensor(y_data, device=device, requires_grad=True)

    z = op_func(x, y)

    expected_z = op_func(np.array(x_data), np.array(y_data))
    assert z.shape == (2, 2)
    assert_allclose(z, expected_z)

    loss = z.sum()
    loss.backward()

    assert x.grad.shape == (
        2,
        1,
    ), f"Unbroadcasting failed for x. Expected (2, 1), got {x.grad.shape}"
    assert y.grad.shape == (
        1,
        2,
    ), f"Unbroadcasting failed for y. Expected (1, 2), got {y.grad.shape}"

    assert_allclose(x.grad, expected_grad_x)
    assert_allclose(y.grad, expected_grad_y)

def test_backward_through_empty_broadcast(device, assert_allclose):
    singleton = dt.Tensor([2.0], device=device, requires_grad=True)
    empty = dt.Tensor(np.empty((0,)), device=device)

    result = singleton + empty
    assert result.shape == (0,)

    result.sum().backward()
    assert singleton.grad.shape == (1,)
    assert_allclose(singleton.grad, [0.0])

@pytest.mark.parametrize("operation, expected", [(dt.add, 100000.0), (dt.subtract, -100000.0)])
def test_broadcast_backward_uses_wider_leaf_dtype(device, assert_allclose, operation, expected):
    leaf = dt.ones((1,), dtype=np.float64, device=device, requires_grad=True)
    other = dt.ones((2, 50000), dtype=np.float64, device=device)
    out = dt.zeros(other.shape, dtype=np.float16, device=device).data

    result = operation(other, leaf, out=out)
    result.backward()

    assert result.dtype == np.float16
    assert leaf.grad.dtype == np.float64
    assert_allclose(leaf.grad, [expected], rtol=0, atol=0)

def test_broadcast_backward_preserves_subnormal_seed_when_widening(device, assert_allclose):
    leaf = dt.ones((), dtype=np.float64, device=device, requires_grad=True)
    other = dt.ones((2,), dtype=np.float64, device=device)
    out = dt.zeros(other.shape, dtype=np.float32, device=device).data
    subnormal = np.nextafter(np.float32(0), np.float32(1))
    seed = dt.Tensor(np.full(other.shape, subnormal, dtype=np.float32), device=device)

    dt.add(other, leaf, out=out).backward(seed)

    assert_allclose(leaf.grad, 2 * float(subnormal), rtol=0, atol=0)
