import operator

import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize(
    "alias", [lambda x: x[:], lambda x: x.transpose(None), dt.Tensor],
    ids=["slice", "transpose", "tensor-wrapper"],
)
def test_mutating_alias_invalidates_saved_values(device, alias):
    x = dt.Tensor([[2.0, 5.0]], device=device, requires_grad=True)
    loss = (x * x).sum()

    alias(x)[0, 0] = 7.0

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

@pytest.mark.parametrize("view", [False, True], ids=["same-array", "array-view"])
def test_mutating_independently_wrapped_storage_invalidates_autograd(device, view):
    buffer = dt.Tensor([[2.0, 5.0], [7.0, 9.0]], device=device).data
    x = dt.Tensor(buffer, requires_grad=True)
    alias = dt.Tensor(buffer.T if view else buffer)
    loss = (x * x).sum()

    alias[0, 1] = 3.0

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

@pytest.mark.parametrize(
    "index", [0, (Ellipsis, 0), slice(0, 1), [0], np.array([True, False, False]), np.array(0)],
)
@pytest.mark.parametrize("array_view", [True, False])
def test_disjoint_cpu_indexed_write_preserves_tracked_view(index, array_view, assert_allclose):
    buffer = np.array([-9.0, 2.0, 3.0])
    alias = dt.Tensor(buffer)
    leaf = dt.Tensor(buffer[1:]) if array_view else alias[1:]
    leaf.toggle_grad(True)
    loss = leaf.square().sum()

    alias[index] = 7.0

    loss.backward()

    assert_allclose(buffer, [7.0, 2.0, 3.0])
    assert_allclose(leaf.grad, [4.0, 6.0])
    assert leaf._version == 0
    assert alias._version == 1

def test_indexed_cpu_write_checks_only_selected_elements_of_active_result(assert_allclose):
    buffer = np.array([-9.0, 0.0, 0.0])
    alias = dt.Tensor(buffer)
    x = dt.Tensor([2.0, 3.0], requires_grad=True)
    result = dt.square(x, out=buffer[1:])
    loss = result.sum()

    alias[[0, 0]] = [5.0, 7.0]
    alias[[]] = 10.0

    with pytest.raises(RuntimeError, match="In-place indexed assignment"):
        alias[[0, 2]] = [4.0, 5.0]

    loss.backward()

    assert_allclose(buffer, [7.0, 4.0, 9.0])
    assert_allclose(x.grad, [4.0, 6.0])
    assert alias._version == 2

def test_indexed_cpu_write_tracks_object_slot_containing_empty_array():
    buffer = np.empty(2, dtype=object)
    buffer[0] = np.empty(0)
    alias = dt.Tensor(buffer)
    selected = dt.Tensor(buffer[:1])
    other = dt.Tensor(buffer[1:])

    alias[np.array(0)] = "updated"
    alias[[]] = "unused"

    assert selected.data[0] == "updated"
    assert alias._version == selected._version == 1
    assert other._version == 0

@pytest.mark.parametrize("value,error", [([9.0, "bad"], ValueError), ([9.0, 10**1000], OverflowError)])
def test_failed_cpu_assignment_leaves_data_and_version_intact(value, error, assert_allclose):
    x = dt.Tensor([1.0, 2.0], requires_grad=True)
    alias = dt.Tensor(x.data)
    loss = x.square().sum()
    version = x._version
    alias_version = alias._version

    with pytest.raises(error):
        x[:] = value

    assert_allclose(x, [1.0, 2.0])
    assert_allclose(alias, [1.0, 2.0])
    assert x._version == version
    assert alias._version == alias_version

    loss.backward()
    assert_allclose(x.grad, [2.0, 4.0])

def test_data_replacement_invalidates_old_graph_and_preserves_old_alias(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    alias = dt.Tensor(x, requires_grad=True)
    loss = x.square().sum()
    alias_loss = alias.square().sum()
    x.one_grad()
    version = x._version

    x.data = dt.Tensor([7.0, 8.0], device=device).data

    assert x._version == version + 1
    assert alias._version == version
    assert x.grad is None

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

    alias_loss.backward()
    assert_allclose(alias.grad, [4.0])

    x.square().sum().backward()
    assert_allclose(x.grad, [14.0, 16.0])

def test_augmented_data_assignment_invalidates_same_storage(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    alias = dt.Tensor(x.data)
    loss = x.square().sum()
    x.one_grad()

    alias.data -= 2 * x.grad

    assert_allclose(x, [0.0])
    assert_allclose(x.grad, [1.0])

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

def test_augmented_data_assignment_invalidates_active_result(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    result = x.square()
    saved = result.copy(copy_prev=True, copy_requires_grad=True)
    alias = dt.Tensor(result.data)

    alias.data *= 2.0

    assert_allclose(result, [8.0])

    copied_after_write = result.copy(copy_prev=True, copy_requires_grad=True)
    for loss in (result, result.sum(), copied_after_write):
        with pytest.raises(RuntimeError, match="modified by an in-place operation"):
            loss.backward()

    assert x.grad is None

    saved.backward()
    assert_allclose(x.grad, [4.0])

def test_data_replacement_checks_tracking_before_mutation(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    y = x.square()

    with pytest.raises(RuntimeError, match="floating-point dtype"):
        x.data = dt.Tensor([1], device=device).data

    with pytest.raises(RuntimeError, match="Cannot replace data"):
        y.data = dt.Tensor([7.0, 8.0], device=device).data

    assert_allclose(x, [2.0])
    assert_allclose(y, [4.0])

    y.backward()
    assert_allclose(x.grad, [4.0])

@pytest.mark.parametrize("array_view", [False, True], ids=["tensor-view", "array-view"])
def test_rebinding_view_does_not_invalidate_original_storage(device, array_view, assert_allclose):
    x = dt.Tensor([2.0, 5.0], device=device, requires_grad=True)
    loss = (x * x).sum()
    view = dt.Tensor(x.data[:] if array_view else x[:])
    view *= 2.0
    view[0] = 7.0

    loss.backward()

    assert_allclose(view, [7.0, 10.0])
    assert_allclose(x, [2.0, 5.0])
    assert_allclose(x.grad, [4.0, 10.0])

@pytest.mark.parametrize(
    "forward",
    [lambda x: x[:], lambda x: x[[1, 0]], dt.Tensor.transpose, dt.Tensor.contiguous],
    ids=["slice", "gather", "transpose", "contiguous"],
)
@pytest.mark.parametrize("rebind", [False, True], ids=["relu", "rebind"])
def test_view_backward_checks_source_version(device, forward, rebind):
    source = dt.Tensor([[-2.0, 5.0], [3.0, -7.0]], device=device).transpose()
    source.toggle_grad(True)
    assert not source.is_contiguous()

    result = forward(source)
    loss = result.sum()

    if rebind:
        source.toggle_grad(False)
        source.mean(in_place=True)
        source.toggle_grad(True)
    else:
        dt.relu(source, in_place=True)

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

    assert source.grad is result.grad is loss.grad is None

@pytest.mark.parametrize(
    "mutate", [operator.imul, lambda x, scale: dt.multiply(x, scale, in_place=True)],
    ids=["operator", "function"],
)
def test_inplace_multiply_rejects_gradient_tracking_destination(device, assert_allclose, mutate):
    x = dt.Tensor([1.0, 2.0], device=device, requires_grad=True)
    y = x * 2.0

    with pytest.raises(RuntimeError, match="[Ii]n-place"):
        mutate(y, 3.0)

    assert_allclose(y, [2.0, 4.0])

    y.sum().backward()
    assert_allclose(x.grad, [2.0, 2.0])

@pytest.mark.parametrize(
    "alias",
    [lambda y: y, lambda y: y[1:], lambda y: dt.Tensor(y.data[1:])],
    ids=["result", "view", "untracked-view"],
)
def test_setitem_rejects_overwriting_differentiable_result(device, alias, assert_allclose):
    x = dt.Tensor([2.0, 5.0], device=device, requires_grad=True)
    result = x * 3.0
    destination = alias(result)
    version = result._version

    with pytest.raises(RuntimeError, match="In-place indexed assignment"):
        destination[0] = 100.0

    assert result._version == version
    assert_allclose(result, [6.0, 15.0])

    result.sum().backward()
    assert_allclose(x.grad, [3.0, 3.0])

def test_repeated_indices_accumulate_gradients(device, assert_allclose):
    seed = [1.0, 2.0, 3.0]
    grad = dt.Tensor(seed, device=device)
    x = dt.Tensor([2.0, 3.0, 5.0], device=device, dtype=grad.dtype, requires_grad=True)

    x[[0, 0, 2]].backward(grad)

    assert_allclose(x.grad, [seed[0] + seed[1], 0.0, seed[2]])

def test_transpose_negative_axes_backward(device, assert_allclose):
    x = dt.Tensor(np.zeros((2, 3, 4)), device=device, requires_grad=True)
    seed = np.arange(24, dtype=np.float64).reshape(4, 2, 3)

    x.transpose((-1, 0, 1)).backward(dt.Tensor(seed, device=device))

    assert_allclose(x.grad, seed.transpose(1, 2, 0))

def test_out_of_bounds_advanced_indices_fail_reads_and_writes(device, assert_allclose):
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    x = dt.Tensor(data, device=device, requires_grad=True)
    mask = dt.Tensor([[True, False, False], [False, True, False]], device=device)
    loss = (x * x).sum()

    for index, axis, size in (
        ([-3], 0, 2),
        (dt.Tensor([2], dtype=np.uint8, device=device).data, 0, 2),
        (dt.Tensor(2, device=device), 0, 2),
        ([dt.Tensor(2, device=device)], 0, 2),
        ([dt.Tensor(-3, device=device)], 0, 2),
        ((None, Ellipsis, [-5]), 2, 4),
        ((mask, [4, 0]), 2, 4),
    ):
        message = f"out of bounds for axis {axis} with size {size}"

        with pytest.raises(IndexError, match=message):
            x[index]

        with pytest.raises(IndexError, match=message):
            x[index] = -1.0

    assert_allclose(x, np.arange(24, dtype=float).reshape(2, 3, 4))

    loss.backward()
    assert_allclose(x.grad, 2 * data)

def test_advanced_indices_preserve_negative_bounds_masks_and_empty_indices(device, assert_allclose):
    data = np.arange(24).reshape(2, 3, 4)
    x = dt.Tensor(data, device=device)
    mask = np.array([[True, False, False], [False, True, False]])
    indices = (None, Ellipsis, [-4, -1])

    assert_allclose(x[dt.Tensor(mask, device=device)], data[mask])
    assert_allclose(x[indices], data[indices])

    tensor_indices = [dt.Tensor(-4, device=device), dt.Tensor(-1, device=device)]
    assert_allclose(x[None, Ellipsis, tensor_indices], data[indices])
    assert x[[]].shape == (0, 3, 4)

    empty = dt.Tensor(np.empty((0, 3)), device=device)
    assert empty[[]].shape == (0, 3)

    with pytest.raises(IndexError):
        empty[[0]]

    with pytest.raises(IndexError):
        empty[[0]] = 1
