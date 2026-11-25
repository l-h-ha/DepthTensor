import numpy as np
import pytest

import depthtensor as dt

def test_numpy_left_operators_return_tensors(assert_allclose):
    x = dt.Tensor([4.0, 5.0], requires_grad=True)
    array = np.array([2.0, 3.0])
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    product = array * x

    for result, expected in (
        (array > x, [False, False]),
        (array - x, [-2.0, -2.0]),
        (array ** x, [16.0, 243.0]),
        (product, [8.0, 15.0]),
        (matrix @ x, [14.0, 32.0]),
    ):
        assert isinstance(result, dt.Tensor)
        assert result.device == "cpu"
        assert_allclose(result, expected)

    product.sum().backward()
    assert_allclose(x.grad, array)

def test_tensor_truth_requires_exactly_one_element(device):
    assert not bool(dt.Tensor(False, device=device))
    assert not bool(dt.Tensor([[0]], device=device))
    assert bool(dt.Tensor([1.0], device=device))

    for data in ([], [True, False]):
        with pytest.raises(ValueError, match="truth value.*ambiguous"):
            bool(dt.Tensor(data, device=device))

def test_transfer_backward_rejects_modified_source(device):
    source = dt.Tensor([-1.0, 2.0], requires_grad=True)
    transferred = source.to_device(device, clear_prev=False)
    loss = transferred.sum()

    dt.relu(source, in_place=True)

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

    assert source.grad is transferred.grad is loss.grad is None

@pytest.mark.parametrize("preseeded", [False, True])
def test_toggle_grad_detaches_from_existing_graph(device, preseeded, assert_allclose):
    x = dt.Tensor([1.0, 2.0], device=device, requires_grad=True)
    detached = x * 2.0
    view = detached[:1]
    seed = dt.Tensor([3.0, 4.0], device=device).data

    if preseeded:
        detached.one_grad()

    detached.toggle_grad(False)
    assert not detached.requires_grad
    assert detached.grad is detached.grad_fn is None
    assert detached.prev == ()

    with pytest.raises(RuntimeError, match="undifferentiable tensor"):
        detached.backward()

    with pytest.raises(RuntimeError, match="undifferentiable tensor"):
        detached.backward(seed)

    view.backward()
    assert x.grad is detached.grad is None

    detached.toggle_grad(True)
    detached.backward(seed)
    assert_allclose(detached.grad, seed)
    assert x.grad is None

@pytest.mark.parametrize("explicit_seed", [False, True])
def test_failed_backward_restores_all_gradient_buffers(device, explicit_seed, assert_allclose):
    bad_leaf = dt.Tensor([2.0], device=device, requires_grad=True)
    coefficient = dt.Tensor([3.0], device=device)
    bad_branch = bad_leaf * coefficient

    good_leaf = dt.Tensor([4.0], device=device, requires_grad=True)
    fresh_leaf = dt.Tensor([7.0], device=device, requires_grad=True)
    good_branch = good_leaf * 5.0 + fresh_leaf
    loss = bad_branch + good_branch

    previous_gradients = []
    for tensor, value in [(loss, 2.0), (good_branch, 3.0), (good_leaf, 4.0)]:
        tensor.zero_grad()
        tensor.grad.fill(value)
        previous_gradients.append((tensor, tensor.grad, value))

    coefficient[0] = 8.0
    seed = dt.Tensor([6.0], device=device).data if explicit_seed else None

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward(seed)

    for tensor, buffer, value in previous_gradients:
        assert tensor.grad is buffer
        assert_allclose(tensor.grad, [value])

    assert fresh_leaf.grad is bad_branch.grad is bad_leaf.grad is None

def test_backward_seed_uses_tensor_dtype_and_independent_storage(device, assert_allclose):
    x = dt.Tensor([2.0], dtype=np.float32, device=device, requires_grad=True)
    seed = dt.Tensor([1], dtype=np.int64, device=device).data

    x.backward(seed)
    (x * 0.5).sum().backward()

    assert x.grad.dtype == x.dtype
    assert_allclose(x.grad, [1.5])
    assert_allclose(seed, [1])

@pytest.mark.parametrize(
    "method,args",
    [
        ("abs", ()), ("clip", (0.0, 3.0)), ("exp", ()), ("log", ()),
        ("max", ()), ("maximum", (0.0,)), ("mean", ()), ("sign", ()),
        ("sqrt", ()), ("square", ()), ("sum", ()),
    ],
)
def test_tensor_methods_detach_with_requires_grad_false(method, args, device):
    x = dt.Tensor([1.0, 2.0], device=device, requires_grad=True)
    operation = getattr(x, method)
    assert operation(*args).requires_grad

    detached = operation(*args, requires_grad=False)
    assert not detached.requires_grad
    assert detached.grad is detached.grad_fn is None
    assert detached.prev == ()

@pytest.mark.parametrize("dtype", [np.bool_, np.int64])
def test_integer_and_boolean_gradients_are_rejected_consistently(device, dtype):
    untracked = dt.Tensor([1, 0], device=device, dtype=dtype)

    with pytest.raises(RuntimeError, match="Gradients require a floating-point dtype"):
        dt.Tensor(untracked.data, requires_grad=True)

    with pytest.raises(RuntimeError, match="Gradients require a floating-point dtype"):
        dt.Tensor._fast_init(untracked.data, device, requires_grad=True)

    with pytest.raises(RuntimeError, match="Gradients require a floating-point dtype"):
        untracked.toggle_grad(True)

    assert not untracked.requires_grad
    assert untracked.grad is untracked.grad_fn is None
