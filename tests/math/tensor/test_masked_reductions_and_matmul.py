import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize("method", ["sum", "mean"])
@pytest.mark.parametrize("axis, keepdims", [(-1, False), ((0, -1), True)])
def test_masked_reduction(device, assert_allclose, method, axis, keepdims):
    data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    data[0, 0, 1] = np.nan
    mask = np.array([[[True, False, True, False]], [[False, True, False, False]]])
    x = dt.Tensor(data, device=device, requires_grad=True)
    where = dt.Tensor(mask.copy(), device=device).data

    result = getattr(x, method)(axis=axis, keepdims=keepdims, where=where)
    expected = getattr(np, method)(data, axis=axis, keepdims=keepdims, where=mask)
    assert_allclose(result, expected)

    seed = np.arange(1, result.size + 1, dtype=np.float64).reshape(result.shape)
    grad = seed if keepdims else np.expand_dims(seed, axis)
    if method == "mean":
        count = np.broadcast_to(mask, data.shape).sum(axis=axis, keepdims=True)
        grad = grad / count

    where[...] = False
    result.backward(dt.Tensor(seed, device=device).data)
    assert_allclose(x.grad, np.where(mask, grad, 0.0))

def test_masked_sum_with_initial_and_output(device, assert_allclose):
    x = dt.Tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    mask = dt.Tensor([True, False], device=device).data
    out = dt.zeros((2,), device=device).data

    result = x.sum(axis=-1, where=mask, initial=10.0, out=out)

    assert result.data is out
    assert_allclose(result, [11.0, 13.0])

@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_sum_applies_initial_before_narrowing_partial_sum(device, dtype):
    x = dt.Tensor([60000.0, 60000.0], dtype=dtype, device=device)
    out = dt.zeros((), dtype=np.float16, device=device).data

    result = x.sum(initial=-60000.0, out=out)

    assert result.data is out
    assert result.item() == 60000.0

def test_sum_with_initial_uses_float16(device):
    x = dt.Tensor([4097.0, -4096.0], dtype=np.float32, device=device)

    result = x.sum(initial=0.0, dtype=np.float16)

    assert result.dtype == np.float16
    assert result.item() == 0.0

def test_mean_with_every_element_masked(device, assert_allclose):
    x = dt.Tensor([1.0, 2.0], device=device, requires_grad=True)
    with np.errstate(invalid="ignore"):
        if device == "cpu":
            with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
                result = x.mean(where=False)
        else:
            result = x.mean(where=False)

    assert np.isnan(result.item())
    result.backward()
    assert_allclose(x.grad, [0.0, 0.0])

@pytest.mark.parametrize(
    "input_dtype, axis",
    [(np.float16, None), (np.float64, -1)],
    ids=["float16-mean", "float16-out"],
)
def test_large_mean_backward_preserves_float16_seed(device, assert_allclose, input_dtype, axis):
    count = 100_000
    shape = (count,) if axis is None else (2, count)
    x = dt.ones(shape, dtype=input_dtype, device=device, requires_grad=True)
    out = None if axis is None else dt.zeros((2,), dtype=np.float16, device=device).data
    result = x.mean(axis=axis, out=out)

    assert result.dtype == np.float16
    assert_allclose(result, np.ones(result.shape))

    result.backward()

    assert_allclose(x.grad, np.full(shape, 1 / count, dtype=input_dtype), rtol=1e-12, atol=0)

@pytest.mark.parametrize("masked", [False, True])
def test_mean_backward_uses_input_precision_with_narrow_out(device, assert_allclose, masked):
    x = dt.ones((3 if masked else 2,), dtype=np.float64, device=device, requires_grad=True)
    out = dt.zeros((), dtype=np.float32, device=device).data
    where = dt.Tensor([True, True, False], device=device).data if masked else True
    result = x.mean(out=out, where=where)
    seed = np.nextafter(np.float32(0), np.float32(1))

    result.backward(dt.Tensor(seed, device=device))

    expected = [float(seed) / 2] * 2
    if masked:
        expected.append(0.0)
    assert_allclose(x.grad, expected, rtol=0, atol=0)

@pytest.mark.parametrize("with_out", [False, True])
def test_mean_explicit_float16_keeps_large_counts(device, with_out):
    x = dt.ones((2, 256, 256), dtype=np.float16, device=device) * np.float16(0.5)
    out = dt.zeros((2, 1, 1), dtype=np.float16, device=device).data if with_out else None

    result = x.mean(axis=(1, 2), dtype=np.float16, keepdims=True, out=out)

    assert result.dtype == np.float16
    assert result.data.tolist() == [[[0.5]], [[0.5]]]
    if with_out:
        assert result.data is out

def _matmul_gradient(operands, index, seed):
    # Basis vectors give the matmul derivatives.
    args = list(operands)
    basis = np.zeros_like(args[index])
    args[index] = basis
    gradient = np.empty_like(basis)
    for position in np.ndindex(basis.shape):
        basis[position] = 1.0
        gradient[position] = np.sum(np.matmul(*args) * seed)
        basis[position] = 0.0
    return gradient

@pytest.mark.parametrize(
    "shapes", [((3,), (3,)), ((3,), (2, 3, 4)), ((2, 4, 3), (3,))]
)
def test_matmul_backward_with_vectors(device, assert_allclose, shapes):
    inputs = [
        np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
        for shape in shapes
    ]
    lhs, rhs = [dt.Tensor(x, device=device, requires_grad=True) for x in inputs]
    expected = np.matmul(*inputs)
    result = lhs @ rhs
    assert_allclose(result, expected)

    seed = np.arange(1, expected.size + 1, dtype=np.float64).reshape(expected.shape)
    result.backward(dt.Tensor(seed, device=device).data)

    assert_allclose(lhs.grad, _matmul_gradient(inputs, 0, seed))
    assert_allclose(rhs.grad, _matmul_gradient(inputs, 1, seed))
