import numpy as np
import pytest
import depthtensor as dt

@pytest.mark.parametrize(
    "method, axis",
    [("sum", 0), ("sum", -1), ("sum", ()), ("max", 0), ("max", -1), ("max", ()), ("mean", ())],
)
def test_scalar_reduction_axis(device, assert_allclose, method, axis):
    data = np.array(2.0)
    x = dt.Tensor(data, device=device, requires_grad=True)
    result = getattr(x, method)(axis=axis, keepdims=True)

    assert result.shape == ()
    assert_allclose(result, getattr(np, method)(data, axis=axis, keepdims=True))

    result.backward()
    assert_allclose(x.grad, np.array(1.0))

@pytest.mark.parametrize("method", ["sum", "max", "mean"])
@pytest.mark.parametrize("axis", [(0,), (-1,), 1, -2])
def test_scalar_reduction_rejects_out_of_bounds_axis(device, method, axis):
    x = dt.Tensor(2.0, device=device)

    with pytest.raises(IndexError):
        getattr(x, method)(axis=axis)

@pytest.mark.parametrize("axis", [0, -1])
def test_scalar_mean_rejects_integer_axis(device, axis):
    x = dt.Tensor(2.0, device=device)

    with pytest.raises(IndexError):
        x.mean(axis=axis)

@pytest.mark.parametrize("method", ["sum", "max", "mean"])
def test_reduction_rejects_duplicate_axis(device, method):
    x = dt.ones((2, 3), device=device)

    with pytest.raises(ValueError):
        getattr(x, method)(axis=(0, -2))

@pytest.mark.parametrize("mask", [False, np.array(True), [True, False, True], [True, True, True]])
def test_masked_sum_distinguishes_none_from_omitted_initial(device, assert_allclose, mask):
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    x = dt.Tensor(data, device=device)
    where = False if mask is False else dt.Tensor(mask, device=device).data
    out = dt.Tensor([-1.0, -1.0], device=device).data

    with pytest.raises(ValueError):
        x.sum(axis=1, where=where, initial=None, out=out)
    assert_allclose(out, [-1.0, -1.0])

    result = x.sum(axis=1, where=where, out=out)
    assert result.data is out
    assert_allclose(result, np.sum(data, axis=1, where=mask))

@pytest.mark.parametrize("shape", [(0, 3), (0, 0)])
def test_sum_none_initial_rejects_empty_reduced_axis(device, assert_allclose, shape):
    data = np.empty(shape)
    x = dt.Tensor(data, device=device)

    with pytest.raises(ValueError):
        x.sum(axis=0, initial=None)

    assert_allclose(x.sum(axis=0), np.sum(data, axis=0))

def test_sum_none_initial_accepts_empty_output(device, assert_allclose):
    data = np.empty((0, 3))
    x = dt.Tensor(data, device=device)
    result = x.sum(axis=1, initial=None)
    assert result.shape == (0,)
    assert_allclose(result, np.sum(data, axis=1, initial=None))

@pytest.mark.parametrize("kwargs", [{}, {"initial": None}], ids=["omitted", "none"])
def test_max_without_initial_validates_empty_reduced_axes(device, assert_allclose, kwargs):
    data = dt.zeros((0, 3), device=device)

    with pytest.raises(ValueError, match="zero-size"):
        dt.zeros((0, 0), device=device).max(axis=0, **kwargs)

    result = data.max(axis=1, **kwargs)

    assert result.shape == (0,)
    assert_allclose(result, np.max(np.empty((0, 3)), axis=1, **kwargs))

@pytest.mark.parametrize("method", ["sum", "mean", "max"])
def test_reduction_rejects_mask_that_cannot_broadcast(device, method):
    x = dt.ones((2, 3), device=device)
    where = dt.ones((2, 2), dtype=bool, device=device).data
    kwargs = {"initial": 0} if method == "max" else {}

    with pytest.raises(ValueError):
        getattr(x, method)(where=where, **kwargs)

def test_sum_none_initial_accepts_nonempty_unmasked_input(device, assert_allclose):
    x = dt.Tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    result = x.sum(axis=-1, initial=None)

    result.backward(dt.Tensor([1.0, 2.0], device=device))

    assert_allclose(result, [3.0, 7.0])
    assert_allclose(x.grad, [[1.0, 1.0], [2.0, 2.0]])
