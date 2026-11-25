import sys

import numpy as np
import pytest

import depthtensor as dt

def test_default_backward_replaces_scalar_seed(device, assert_allclose):
    x = dt.Tensor(2.0, device=device, requires_grad=True)
    result = x * 2.0
    seed = np.float64(3.0) if device == "cpu" else dt.Tensor(3.0, device=device).data

    result.backward(seed)
    assert_allclose(result.grad, 3.0)

    result.backward()
    assert_allclose(result.grad, 1.0)
    assert_allclose(x.grad, 8.0)

def test_backward_on_graph_deeper_than_recursion_limit(device, assert_allclose):
    x = dt.Tensor(2.0, device=device, requires_grad=True)
    result = x
    depth = sys.getrecursionlimit() + 25

    for _ in range(depth):
        result = result + 1.0

    topology = result.backward()

    assert len(topology) == depth + 1
    assert topology[0] is x
    assert topology[-1] is result
    assert_allclose(x.grad, 1.0)

def test_backward_keeps_parent_order_with_shared_ancestor(device, assert_allclose):
    x = dt.Tensor(2.0, device=device, requires_grad=True)
    left = x * 2.0
    right = x * 3.0
    result = left + right

    topology = result.backward()

    assert [id(tensor) for tensor in topology] == [id(x), id(left), id(right), id(result)]
    assert_allclose(x.grad, 5.0)

def test_backward_detects_cycles_before_accumulating_gradients(device):
    x = dt.Tensor(2.0, device=device, requires_grad=True)
    result = x * 2.0
    x.prev = (result,)

    with pytest.raises(dt.GradientComputationError, match="cycle"):
        result.backward()

    assert x.grad is None
    assert result.grad is None

def test_tensor_indices_support_reads_and_writes(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0, 5.0], device=device)
    indices = dt.Tensor([0, 2], device="cpu")
    mask = dt.Tensor([False, True, True], device=device)

    assert_allclose(x[indices], [2.0, 5.0])
    assert_allclose(x[indices[1]], 5.0)

    x[indices] = 7.0
    assert_allclose(x[mask], [3.0, 7.0])

    x[mask] = -1.0
    assert_allclose(x, [7.0, -1.0, -1.0])

def test_backward_uses_saved_tensor_index_tuple(device, assert_allclose):
    x = dt.Tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]], device=device, requires_grad=True)
    rows = dt.Tensor([0, 0, 2], device=device)
    columns = dt.Tensor([1, 1, 0], device=device)
    selected = x[rows, columns]

    assert_allclose(selected, [3.0, 3.0, 11.0])

    rows[:] = 1
    columns[:] = 0

    selected.sum().backward()
    assert_allclose(x.grad, [[0.0, 2.0], [0.0, 0.0], [1.0, 0.0]])

    x[rows, columns] = -1.0
    assert_allclose(x, [[2.0, 3.0], [-1.0, 7.0], [11.0, 13.0]])

def test_backward_uses_saved_slice_bounds(device, assert_allclose):
    x = dt.Tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    start, stop = np.array(0), np.array(2)
    loss = x[slice(start, stop)].square().sum()

    start[...] = 1
    stop[...] = 3

    loss.backward()

    assert_allclose(loss, 5.0)
    assert_allclose(x.grad, [2.0, 4.0, 0.0])

def test_backward_uses_saved_slice_step(device, assert_allclose):
    x = dt.Tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    step = np.array(-1)
    selected = x[..., ::step]
    loss = selected.square().sum()

    step[...] = 1

    loss.backward()

    assert_allclose(selected, [3.0, 2.0, 1.0])
    assert_allclose(x.grad, [2.0, 4.0, 6.0])

def test_backward_uses_saved_scalar_index(device, assert_allclose):
    class MutableIndex:
        value = 0

        def __index__(self):
            return self.value

    x = dt.Tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    index = MutableIndex()
    loss = x[index].square().sum()

    index.value = 1

    loss.backward()

    assert_allclose(loss, 5.0)
    assert_allclose(x.grad, [[2.0, 4.0], [0.0, 0.0]])

@pytest.mark.parametrize(
    "operation, expected",
    [
        ("sum", [[1.0, 10.0], [1.0, 10.0]]),
        ("mean", [[0.5, 5.0], [0.5, 5.0]]),
        ("max", [[0.0, 10.0], [1.0, 0.0]]),
    ],
)
def test_backward_uses_saved_reduction_axis(device, assert_allclose, operation, expected):
    x = dt.Tensor([[1.0, 4.0], [3.0, 2.0]], device=device, requires_grad=True)
    axis = np.array(0)
    result = getattr(x, operation)(axis=(axis,))

    axis[...] = 1

    result.backward(dt.Tensor([1.0, 10.0], device=device))

    assert_allclose(x.grad, expected)

@pytest.mark.parametrize(
    "operation, expected",
    [
        ("sum", [[1.0, 1.0], [10.0, 10.0]]),
        ("mean", [[0.5, 0.5], [5.0, 5.0]]),
        ("max", [[0.0, 1.0], [10.0, 0.0]]),
    ],
)
def test_backward_uses_saved_keepdims(device, assert_allclose, operation, expected):
    x = dt.Tensor([[1.0, 4.0], [3.0, 2.0]], device=device, requires_grad=True)
    keepdims = np.array(0)
    result = getattr(x, operation)(axis=1, keepdims=keepdims)

    keepdims[...] = 1

    result.backward(dt.Tensor([1.0, 10.0], device=device))

    assert_allclose(x.grad, expected)

def test_to_device_preserves_gradients_through_leaf_and_intermediate(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    leaf_copy = x.to_device("cpu", clear_prev=False)
    intermediate = leaf_copy * leaf_copy
    result = intermediate.to_device(device, clear_prev=False)
    loss = result.sum()

    assert leaf_copy is not x
    assert leaf_copy.device == "cpu"
    assert result.device == device
    assert leaf_copy.requires_grad
    assert result.requires_grad
    assert_allclose(result, [4.0, 9.0])

    loss.backward()
    assert_allclose(intermediate.grad, [1.0, 1.0])
    assert_allclose(x.grad, [4.0, 6.0])

    loss.backward()
    assert_allclose(x.grad, [8.0, 12.0])

def test_to_device_detaches_by_default(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    source = x * x
    detached = source.to_device("cpu")

    assert not detached.requires_grad
    assert detached.prev == ()
    assert detached.grad_fn is None
    assert_allclose(detached, [4.0, 9.0])
    assert source.to_device(device, in_place=True) is source
