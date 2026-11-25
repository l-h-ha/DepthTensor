import numpy as np
import pytest

import depthtensor as dt

def test_copy_with_inplace_relu_history_preserves_the_mask(device, assert_allclose):
    original = dt.Tensor([-2.0, 5.0], device=device, requires_grad=True)
    dt.relu(original, in_place=True)
    copied = original.copy(copy_prev=True, copy_requires_grad=True, dtype=np.float32)

    loss = copied.sum()
    loss.backward()
    assert_allclose(copied.grad, [0.0, 1.0])
    assert copied.grad.dtype == copied.dtype
    assert original.grad is None

    loss.backward()
    assert_allclose(copied.grad, [0.0, 2.0])
    assert original.grad is None

    original.sum().backward()
    assert_allclose(original.grad, [0.0, 1.0])

@pytest.mark.parametrize("copy_to_cpu", [False, True])
def test_copy_grad_has_independent_storage(device, assert_allclose, copy_to_cpu):
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    x.one_grad()

    destination = "cpu" if copy_to_cpu else device
    copied = x.copy(device=destination, dtype=np.float32, copy_grad=True, copy_requires_grad=True)
    assert copied.device == destination
    assert copied.grad.dtype == copied.dtype == np.float32
    assert_allclose(copied.grad, [1.0, 1.0])

    copied.zero_grad()
    assert_allclose(copied.grad, [0.0, 0.0])
    assert_allclose(x.grad, [1.0, 1.0])

def test_copy_prev_with_gradients_disabled_stays_detached(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    copied = (x * 2.0).copy(copy_prev=True)

    assert not copied.requires_grad
    assert copied.prev == ()

    copied.toggle_grad(True)
    copied.sum().backward()

    assert_allclose(copied.grad, [1.0, 1.0])
    assert x.grad is None

def test_copy_backward_preserves_graph_and_source_state(device, assert_allclose):
    x = dt.Tensor([[2.0, -1.0], [-3.0, 3.0]], device=device, requires_grad=True)
    original = x.max(axis=1)
    original.one_grad()
    original.grad *= 2

    copied = original.copy(device="cpu", dtype=np.float32, copy_prev=True, copy_requires_grad=True)
    original.data[0] = -17.0

    loss = copied.sum()
    loss.backward()
    assert_allclose(x.grad, [[1.0, 0.0], [0.0, 1.0]])

    loss.backward()
    assert_allclose(x.grad, [[2.0, 0.0], [0.0, 2.0]])
    assert_allclose(original.grad, [2.0, 2.0])
    assert_allclose(original, [-17.0, 3.0])
    assert original.device == device

def test_copy_backward_checks_parent_version(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    original = x * x
    original.one_grad()
    copied = original.copy(copy_prev=True, copy_requires_grad=True)
    x[0] = 7.0

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        copied.sum().backward()

    assert_allclose(original, [4.0, 9.0])
    assert_allclose(original.grad, [1.0, 1.0])

def test_copy_chain_preserves_backward_and_source_state(device, assert_allclose):
    x = dt.Tensor([-2.0, 5.0], device=device, dtype=np.float64, requires_grad=True)
    original = dt.relu(x * 2.0, in_place=True)
    narrow = original.copy(device="cpu", dtype=np.float32, copy_prev=True, copy_requires_grad=True)
    narrow.one_grad()
    narrow_data, narrow_grad = narrow.data, narrow.grad

    copied = narrow.copy(device=device, dtype=np.float64, copy_prev=True, copy_requires_grad=True)

    for _ in range(1198):
        copied = copied.copy(copy_prev=True, copy_requires_grad=True)

    seed = dt.Tensor([3.0, 1.0 + 2**-28], device=device, dtype=np.float64)
    copied.backward(seed)
    assert_allclose(copied.grad, [0.0, 1.0], rtol=0, atol=0)
    assert_allclose(x.grad, [0.0, 2.0], rtol=0, atol=0)
    assert narrow.data is narrow_data and narrow.grad is narrow_grad
    assert narrow.device == "cpu"
    assert original.grad is None

    copied_grad, x_grad = copied.grad, x.grad
    x[0] = -7.0
    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        copied.backward(seed)

    assert copied.grad is copied_grad and x.grad is x_grad
    assert narrow.data is narrow_data and narrow.grad is narrow_grad
    assert narrow.device == "cpu"
    assert original.grad is None

def test_backward_through_copy_and_relu_chain(device, assert_allclose):
    x = dt.Tensor([-2.0, 3.0], device=device, requires_grad=True)
    original = result = x * 2.0

    for _ in range(600):
        result = result.copy(copy_prev=True, copy_requires_grad=True)
        dt.relu(result, in_place=True)

    loss = result.sum()
    loss.backward()
    loss.backward()

    assert_allclose(x.grad, [0.0, 4.0])
    assert_allclose(result.grad, [0.0, 2.0])
    assert_allclose(original, [-4.0, 6.0])
    assert original.grad is None

def test_gather_backward_uses_forward_indices(device, assert_allclose):
    x = dt.Tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]], device=device, requires_grad=True)
    row_indices = dt.Tensor([0, 0, 2], device=device).data
    column_indices = [1, 1, 0]
    gathered = x[row_indices, column_indices]
    assert_allclose(gathered, [3.0, 3.0, 11.0])

    row_indices[:] = 1
    column_indices[:] = [0, 0, 1]

    gathered.sum().backward()
    assert_allclose(x.grad, [[0.0, 2.0], [0.0, 0.0], [1.0, 0.0]])
