import pytest

import depthtensor as dt

def test_functional_arithmetic_infers_tracking(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)

    dt.square(dt.multiply(x, 3.0)).sum().backward()
    assert_allclose(x.grad, [36.0, 54.0])

    detached = dt.multiply(x, 3.0, requires_grad=False)
    assert not detached.requires_grad
    assert detached.prev == ()

def test_out_cannot_overwrite_tracked_input(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    loss = (x * x).sum()

    with pytest.raises(RuntimeError, match="out"):
        dt.add(x, 1.0, out=x.data)

    assert_allclose(x, [2.0])
    loss.backward()
    assert_allclose(x.grad, [4.0])

def test_write_to_untracked_out_alias_invalidates_saved_values(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    scale = dt.Tensor([3.0], device=device)
    loss = (x * scale).sum()

    dt.add(scale, 1.0, out=scale.data)
    assert_allclose(scale, [4.0])

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

def test_in_place_mean_updates_device_metadata(device, assert_allclose):
    pytest.importorskip("cupy")

    target = "cuda" if device == "cpu" else "cpu"
    x = dt.Tensor([1.0, 3.0], device=device)

    y = dt.mean(x, device=target, in_place=True)

    assert y is x
    assert y.device == dt.utils.get_device(y.data) == target
    assert_allclose(y.sum(), 2.0)
