import numpy as np
import pytest

import depthtensor as dt

def test_zero_power_has_zero_base_gradient(device, assert_allclose):
    x = dt.Tensor([0.0, 1.0, 2.0], device=device, requires_grad=True)

    with np.errstate(divide="raise", invalid="raise"):
        result = x ** 0
        result.sum().backward()

    assert_allclose(result, [1.0, 1.0, 1.0])
    assert_allclose(x.grad, [0.0, 0.0, 0.0])

def test_broadcast_power_gradient_with_zero_exponents(device, assert_allclose):
    x = dt.Tensor([[0.0], [2.0]], device=device, requires_grad=True)
    exponents = dt.Tensor([0, 1, 2], device=device)

    with np.errstate(divide="raise", invalid="raise"):
        (x ** exponents).sum().backward()

    assert_allclose(x.grad, [[1.0], [5.0]])

def test_clip_backward_with_broadcast_reversed_bounds(device, assert_allclose):
    x = dt.Tensor([[0.0], [2.5], [4.0]], device=device, requires_grad=True)
    lower = dt.Tensor([3.0, 1.0], device=device, requires_grad=True)
    upper = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)

    result = dt.clip(x, lower, upper)
    result.sum().backward()

    assert_allclose(result, [[2.0, 1.0], [2.0, 2.5], [2.0, 3.0]])
    assert_allclose(x.grad, [[0.0], [1.0], [0.0]])
    assert_allclose(lower.grad, [0.0, 1.0])
    assert_allclose(upper.grad, [3.0, 1.0])

def test_clip_rejects_output_overlapping_tracked_bound(device, assert_allclose):
    x = dt.Tensor([0.0, 2.0], device=device)
    lower = dt.Tensor([1.0, 1.0], device=device, requires_grad=True)

    with pytest.raises(RuntimeError, match="overlaps"):
        dt.clip(x, lower, 3.0, out=lower.data)

    assert_allclose(lower, [1.0, 1.0])

def test_clip_backward_rejects_mutated_untracked_bound(device):
    x = dt.Tensor([0.0, 2.0], device=device, requires_grad=True)
    lower = dt.Tensor([1.0, 1.0], device=device)
    loss = dt.clip(x, lower, 3.0).sum()

    dt.add(lower, 2.0, out=lower.data)

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

@pytest.mark.parametrize("operation", ["sum", "mean", "max"])
def test_reduction_backward_stops_at_untracked_input(operation, device):
    leaf = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    intermediate = leaf * 2.0
    out = getattr(dt, operation)(intermediate)
    intermediate.toggle_grad(False)

    out.backward()

    assert leaf.grad is None
    assert intermediate.grad is None
