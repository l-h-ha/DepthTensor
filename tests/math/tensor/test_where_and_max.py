import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize(
    "condition_values",
    [
        [[True, False, True], [False, True, False]],
        [[2, 0, -5], [0, 3, 0]],
    ],
    ids=["bool", "integer"],
)
def test_where_backward_uses_saved_condition_and_unbroadcasts(
    device, assert_allclose, condition_values
):
    condition = dt.Tensor(condition_values, device=device)
    x = dt.Tensor([[10.0], [20.0]], device=device, requires_grad=True)
    y = dt.Tensor([[1.0, 2.0, 3.0]], device=device, requires_grad=True)
    result = dt.where(condition, x, y)
    assert_allclose(result, [[10.0, 2.0, 10.0], [1.0, 20.0, 3.0]])

    condition.data[...] = False
    seed = dt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device).data
    result.backward(seed)

    assert_allclose(x.grad, [[4.0], [5.0]])
    assert_allclose(y.grad, [[4.0, 2.0, 6.0]])

def test_where_backward_with_scalar_branch(device, assert_allclose):
    condition = dt.Tensor([True, False], device=device)
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    dt.where(condition, x, 0.0).sum().backward()

    assert_allclose(x.grad, [1.0, 0.0])

def test_where_explicit_device_returns_gradients_to_input(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0], requires_grad=True)
    y = dt.Tensor([1.0, 4.0], requires_grad=True)
    result = dt.where([True, False], x, y, device=device)

    assert result.device == device
    assert_allclose(result, [2.0, 4.0])

    result.sum().backward()
    assert isinstance(x.grad, np.ndarray)
    assert isinstance(y.grad, np.ndarray)
    assert_allclose(x.grad, [1.0, 0.0])
    assert_allclose(y.grad, [0.0, 1.0])

def test_masked_comparison_uses_out_and_device(device, assert_allclose):
    x = dt.Tensor([1.0, 2.0, 3.0])
    mask = dt.Tensor([True, False, True], device=device).data
    out = dt.ones((3,), dtype=np.bool_, device=device).data
    result = dt.greater(x, 3.0, out=out, where=mask, device=device)

    assert result.device == device
    assert result.data is out
    assert_allclose(out, [False, True, False])

def test_masked_elementwise_maximum_backward_and_out(device, assert_allclose):
    x = dt.Tensor([[2.0], [4.0]], requires_grad=True)
    y = dt.Tensor([[1.0, 3.0, 2.0]], device=device, requires_grad=True)
    mask = dt.Tensor([[True, True, False], [True, False, True]], device=device).data
    out = dt.Tensor(np.full((2, 3), 99.0), device=device).data
    result = dt.maximum(x, y, out=out, where=mask, device=device)
    assert result.data is out
    assert_allclose(result, [[2.0, 3.0, 99.0], [4.0, 99.0, 4.0]])

    mask[...] = False
    seed = dt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    result.backward(seed)

    assert isinstance(x.grad, np.ndarray)
    assert_allclose(x.grad, [[1.0], [10.0]])
    assert_allclose(y.grad, [[0.0, 2.0, 0.0]])

@pytest.mark.parametrize("keepdims", [True, False])
def test_masked_max_backward_splits_included_ties(device, assert_allclose, keepdims):
    x = dt.Tensor(
        [[2.0, 10.0, 2.0], [3.0, 3.0, 8.0], [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0]],
        device=device,
        requires_grad=True,
    )
    mask = dt.Tensor(
        [[True, False, True], [True, False, False], [False, False, False], [True, True, True]],
        device=device,
    ).data
    result = x.max(axis=-1, keepdims=keepdims, where=mask, initial=0.0)
    expected = np.array([2.0, 3.0, 0.0, 0.0]).reshape(result.shape)
    assert result.shape == ((4, 1) if keepdims else (4,))
    assert_allclose(result, expected)

    mask[...] = False
    seed = dt.Tensor(np.array([8.0, 6.0, 4.0, 2.0]).reshape(result.shape), device=device)
    result.backward(seed)
    assert_allclose(x.grad, [[4.0, 0.0, 4.0], [6.0, 0.0, 0.0], [0.0] * 3, [0.0] * 3])

def test_max_backward_selects_winner_before_output_cast(device, assert_allclose):
    x = dt.Tensor([1.00000001, 1.00000002], device=device, dtype=np.float64, requires_grad=True)
    out = dt.zeros((), dtype=np.float32, device=device).data
    result = x.max(out=out)
    assert result.data is out
    assert result.item() == 1.0

    result.backward()
    assert_allclose(x.grad, [0.0, 1.0])

def test_max_with_initial_across_empty_axis(device, assert_allclose):
    x = dt.Tensor(np.empty((2, 0)), device=device, requires_grad=True)
    result = x.max(axis=1, initial=2.0)
    assert_allclose(result, [2.0, 2.0])

    result.sum().backward()
    assert_allclose(x.grad, np.empty((2, 0)))
