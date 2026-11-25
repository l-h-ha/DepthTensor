import numpy as np
import pytest

import depthtensor as dt
from depthtensor._core.exceptions import DeviceMismatch

def test_backward_uses_saved_array_operand(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    xp = np if device == "cpu" else pytest.importorskip("cupy")
    coefficient = xp.array([3.0])
    loss = (x * coefficient).sum()

    dt.add(coefficient, 2.0, out=coefficient)
    loss.backward()

    assert loss.item() == 6.0
    assert_allclose(coefficient, [5.0])
    assert_allclose(x.grad, [3.0])

@pytest.mark.parametrize("operation", [
    pytest.param(lambda x: x * x, id="multiply"),
    pytest.param(dt.square, id="square"),
    pytest.param(dt.relu, id="relu"),
])
def test_backward_uses_values_saved_before_data_write(
    device, operation, assert_allclose
):
    x = dt.Tensor([-2.0, 3.0], device=device, requires_grad=True)
    loss = operation(x).sum()

    x.data[0] = 7.0
    loss.backward()

    expected = [0.0, 1.0] if operation is dt.relu else [-4.0, 6.0]
    assert_allclose(x.grad, expected)

@pytest.mark.parametrize("operation", [
    pytest.param(dt.maximum, id="maximum"),
    pytest.param(lambda x, bound: dt.clip(x, bound, None), id="clip"),
])
def test_backward_uses_saved_array_bounds(device, operation, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    xp = np if device == "cpu" else pytest.importorskip("cupy")
    bound = xp.array([3.0])
    loss = operation(x, bound).sum()

    bound[...] = 1.0
    loss.backward()

    assert loss.item() == 3.0
    assert_allclose(x.grad, [0.0])

@pytest.mark.parametrize("scalar", [np.int64(2), np.float32(2.0), np.bool_(True)])
def test_numpy_scalars_follow_tensor_device(device, scalar, assert_allclose):
    x = dt.Tensor([3.0, 4.0], device=device)

    for result in (scalar + x, x + scalar):
        assert result.device == device
        assert_allclose(result, [3.0 + float(scalar), 4.0 + float(scalar)])

def test_where_with_scalar_condition_uses_branch_device(device, assert_allclose):
    x = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    y = dt.Tensor([4.0, 5.0], device=device)

    for result in (dt.where(True, x, y), dt.where(np.bool_(False), 0.0, x)):
        assert result.device == device
        assert_allclose(result, [2.0, 3.0])
        result.sum().backward()

    assert_allclose(x.grad, [2.0, 2.0])

def test_where_rejects_conflicting_implicit_devices():
    pytest.importorskip("cupy")

    condition = dt.Tensor([True])
    x = dt.Tensor([3.0], device="cuda")

    with pytest.raises(DeviceMismatch):
        dt.where(condition, x, 0.0)

    assert dt.where(condition, x, 0.0, device="cuda").device == "cuda"

@pytest.mark.parametrize("operation", [
    pytest.param(lambda x, **kw: dt.add(x, 1, **kw), id="add"),
    pytest.param(dt.negative, id="negative"),
    pytest.param(lambda x, **kw: dt.clip(x, 0, 4, **kw), id="clip"),
    pytest.param(lambda x, **kw: dt.maximum(x, 0, **kw), id="maximum"),
    pytest.param(lambda x, **kw: dt.greater(x, 0, **kw), id="greater"),
    pytest.param(lambda x, **kw: dt.matmul(x, x, **kw), id="matmul"),
])
def test_explicit_output_order(device, operation):
    x = dt.Tensor([[1.0, 2.0], [3.0, 4.0]], device=device)

    assert operation(x, order="F").data.flags.f_contiguous
    assert operation(x, order="C").data.flags.c_contiguous

    with pytest.raises(ValueError, match="order"):
        operation(x, order="invalid")

def test_masked_output_order_and_supplied_buffer(device, assert_allclose):
    x = dt.Tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    out = dt.zeros((2, 2), device=device).data
    mask = [[True, False], [False, True]]

    assert dt.add(x, 1, where=mask, order="F").data.flags.f_contiguous

    result = dt.add(x, 1, where=mask, order="F", out=out)

    assert result.data is out
    assert out.flags.c_contiguous
    assert_allclose(result, [[2.0, 0.0], [0.0, 5.0]])
