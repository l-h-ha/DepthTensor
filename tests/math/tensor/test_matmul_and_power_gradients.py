import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize(
    "data, weights, mask, weights_on_left, expected_grad",
    [
        ([[2.0], [np.nan]], [[3.0]], [[True], [False]], False, [[2.0]]),
        ([[2.0, np.nan]], [[3.0]], [[True, False]], True, [[2.0]]),
        (
            [[[2.0], [np.nan]], [[np.nan], [4.0]]],
            [3.0],
            [[True, False], [False, True]],
            False,
            [6.0],
        ),
    ],
    ids=["right-operand", "left-operand", "batched-vector"],
)
def test_matmul_backward_skips_masked_nonfinite_values(
    device, assert_allclose, data, weights, mask, weights_on_left, expected_grad
):
    x = dt.Tensor(data, device=device)
    w = dt.Tensor(weights, device=device, requires_grad=True)
    mask = dt.Tensor(mask, device=device).data
    values = (w @ x) if weights_on_left else (x @ w)
    loss = dt.where(mask, values, 0.0).sum()

    assert_allclose(loss, 3 * np.sum(expected_grad))
    loss.backward()
    assert_allclose(w.grad, expected_grad)

def test_matmul_backward_keeps_active_nonfinite_values(device, assert_allclose):
    x = dt.Tensor([[np.nan], [2.0]], device=device)
    w = dt.Tensor([[3.0, 4.0]], device=device, requires_grad=True)
    mask = dt.Tensor([[False, True], [True, True]], device=device).data

    dt.where(mask, x @ w, 0.0).sum().backward()

    assert_allclose(w.grad, [[2.0, np.nan]])

@pytest.mark.parametrize(
    "operation, value, seed, expected",
    [
        (dt.square, 0.25, 40000.0, 20000.0),
        (lambda x: dt.power(x, np.float16(-0.5)), 0.0001, 0.01, -5000.0),
    ],
    ids=["square", "power"],
)
def test_float16_backward_avoids_intermediate_overflow(
    device, assert_allclose, operation, value, seed, expected
):
    x = dt.Tensor([value], dtype=np.float16, device=device, requires_grad=True)
    upstream = dt.Tensor([seed], dtype=np.float16, device=device).data

    operation(x).backward(upstream)

    assert_allclose(x.grad, [expected], rtol=0, atol=0)

@pytest.mark.parametrize(
    "operation, value, constant, reverse, expected",
    [
        (dt.multiply, 0.0, 500.0, False, 500000.0),
        (dt.divide, 0.0, 2**-8, False, 256000.0),
        (dt.divide, 2**-6, 0.5, True, -2048000.0),
    ],
    ids=["multiply", "divide-numerator", "divide-denominator"],
)
def test_pointwise_backward_widens_narrow_output_seed(
    device, assert_allclose, operation, value, constant, reverse, expected
):
    x = dt.Tensor([value], dtype=np.float64, device=device, requires_grad=True)
    other = dt.Tensor([constant], dtype=np.float16, device=device)
    out = dt.zeros((1,), dtype=np.float16, device=device).data
    seed = dt.Tensor([1000.0], dtype=np.float16, device=device)
    args = (other, x) if reverse else (x, other)

    result = operation(*args, out=out)
    result.backward(seed)

    assert result.data is out
    assert np.isfinite(result.item())
    assert_allclose(x.grad, [expected], rtol=0, atol=0)

@pytest.mark.parametrize("reverse", [False, True], ids=["left", "right"])
def test_matmul_backward_widens_narrow_output_seed(device, assert_allclose, reverse):
    width = 100_000
    shape = (width, 1) if reverse else (1, width)
    x = dt.Tensor([0.0], dtype=np.float64, device=device, requires_grad=True)
    weight = dt.ones(shape, dtype=np.float16, device=device)
    out = dt.zeros((width,), dtype=np.float16, device=device).data
    seed = dt.ones((width,), dtype=np.float16, device=device)
    args = (weight, x) if reverse else (x, weight)

    result = dt.matmul(*args, out=out)
    result.backward(seed)

    assert result.data is out
    assert_allclose(x.grad, [width], rtol=0, atol=0)

@pytest.mark.parametrize(
    "dtype, base",
    [(np.int8, 1.01), (np.int16, 1.0001), (np.int32, 1.000000001), (np.int64, 1.0)],
)
def test_power_backward_handles_minimum_signed_exponents(device, assert_allclose, dtype, base):
    minimum = np.iinfo(dtype).min
    x = dt.Tensor([base, -base], dtype=np.float64, device=device, requires_grad=True)
    exponent = dt.Tensor([minimum, minimum], dtype=dtype, device=device)

    dt.power(x, exponent).sum().backward()

    expected = minimum * base ** (minimum - 1)
    assert_allclose(x.grad, [expected, -expected], rtol=1e-10, atol=0)
