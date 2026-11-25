import numpy as np
import pytest

import depthtensor as dt

def test_log_backward_uses_input_gradient_precision(device, assert_allclose):
    x = dt.Tensor([2**-10], dtype=np.float64, device=device, requires_grad=True)
    seed = dt.Tensor([128], dtype=np.float16, device=device)

    result = dt.log(x, dtype=np.float16)
    result.backward(seed)

    assert result.dtype == np.float16
    assert x.grad.dtype == np.float64
    assert_allclose(x.grad, [131072.0], rtol=0, atol=0)

@pytest.mark.parametrize("out_dtype", [np.float16, np.float64], ids=["narrow", "wide"])
def test_sqrt_backward_precision_preserves_forward_dtype_and_output(
    device, assert_allclose, out_dtype
):
    value = 1.5001
    rounded_value = np.float16(value)
    x = dt.Tensor([value], dtype=np.float64, device=device, requires_grad=True)
    out = dt.zeros((1,), dtype=out_dtype, device=device).data
    seed = dt.Tensor([256], dtype=out_dtype, device=device)

    result = dt.sqrt(x, dtype=np.float16, out=out)
    result.backward(seed)

    assert result.data is out
    assert result.dtype == out_dtype
    assert x.grad.dtype == np.float64
    assert_allclose(result, [np.sqrt(rounded_value)], rtol=0, atol=0)
    assert_allclose(x.grad, [128.0 / np.sqrt(float(rounded_value))], rtol=1e-14, atol=0)

def test_divide_backward_widens_saved_denominator_before_squaring(device, assert_allclose):
    value = 0.7
    rounded_value = np.float16(value)
    denominator = dt.Tensor([value], dtype=np.float64, device=device, requires_grad=True)

    result = dt.divide(1.0, denominator, dtype=np.float16)
    result.backward()

    assert result.dtype == np.float16
    assert denominator.grad.dtype == np.float64
    assert_allclose(result, [np.float16(1.0) / rounded_value], rtol=0, atol=0)
    assert_allclose(denominator.grad, [-1.0 / float(rounded_value)**2], rtol=1e-14, atol=0)

def test_power_backward_with_integer_base_and_float_exponent(device, assert_allclose):
    base = dt.Tensor([2], dtype=np.int8, device=device)
    exponent = dt.Tensor([1.0], dtype=np.float64, device=device, requires_grad=True)

    result = dt.power(base, exponent)
    result.backward()

    assert result.dtype == np.float64
    assert exponent.grad.dtype == np.float64
    assert_allclose(result, [2.0], rtol=0, atol=0)
    assert_allclose(exponent.grad, [2.0 * np.log(2.0)], rtol=1e-14, atol=0)

@pytest.mark.parametrize(
    "previous_grad, expected_grad",
    [(None, np.inf), (-65_504.0, 34_496.0)],
    ids=["fresh", "accumulated"],
)
def test_backward_passes_wide_gradients_through_float16_intermediate(
    device, assert_allclose, previous_grad, expected_grad
):
    x = dt.Tensor([0.25], dtype=np.float64, device=device, requires_grad=True)
    intermediate = x.square(dtype=np.float16)
    weight = dt.Tensor([100_000.0], dtype=np.float64, device=device)
    result = intermediate * weight
    seed = dt.Tensor([0.5], dtype=np.float64, device=device)

    if previous_grad is not None:
        intermediate.zero_grad()
        intermediate.grad.fill(previous_grad)

    with np.errstate(over="ignore"):
        result.backward()
        assert_allclose(x.grad, [50_000.0], rtol=0, atol=0)
        assert_allclose(intermediate.grad, [expected_grad], rtol=0, atol=0)

        result.backward(seed)

    assert intermediate.grad.dtype == np.float16
    assert_allclose(intermediate.grad, [np.inf])
    assert x.grad.dtype == np.float64
    assert_allclose(x.grad, [75_000.0], rtol=0, atol=0)
    assert_allclose(result.grad, [0.5], rtol=0, atol=0)

def test_backward_uses_float64_seed_with_float16_output(device, assert_allclose):
    x = dt.Tensor([0.25], dtype=np.float64, device=device, requires_grad=True)
    result = x.square(dtype=np.float16)
    seed = dt.Tensor([1e100], dtype=np.float64, device=device)

    with np.errstate(over="ignore"):
        result.backward(seed)

    assert result.grad.dtype == np.float16
    assert_allclose(result.grad, [np.inf])
    assert_allclose(seed, [1e100], rtol=0, atol=0)
    assert_allclose(x.grad, [5e99], rtol=1e-15, atol=0)

def test_backward_combines_wide_branches_before_narrowing(device, assert_allclose):
    x = dt.Tensor([0.25], dtype=np.float64, device=device, requires_grad=True)
    intermediate = x.square(dtype=np.float16)
    positive = dt.Tensor([2.0**40], dtype=np.float64, device=device)
    negative = dt.Tensor([1.0 - 2.0**40], dtype=np.float64, device=device)
    result = intermediate * positive + intermediate * negative

    with np.errstate(over="ignore", invalid="ignore"):
        result.backward()

    assert intermediate.grad.dtype == np.float16
    assert_allclose(intermediate.grad, [1.0], rtol=0, atol=0)
    assert_allclose(x.grad, [0.5], rtol=0, atol=0)

def test_backward_passes_wide_gradients_through_views_and_gather(device, assert_allclose):
    x = dt.Tensor([[0.25, 0.5], [0.25, 0.75]], dtype=np.float64, device=device, requires_grad=True)
    intermediate = x.square(dtype=np.float16)
    transposed = intermediate[:, :1].transpose().contiguous()
    transferred = transposed.to_device(device, clear_prev=False)
    selected = transferred[0, [0, 0, 1]]
    weights = dt.Tensor([1e40, -1e40, 1e40], dtype=np.float64, device=device)

    with np.errstate(over="ignore", invalid="ignore"):
        (selected * weights).sum().backward()

    assert intermediate.grad.dtype == np.float16
    assert_allclose(intermediate.grad, [[0.0, 0.0], [np.inf, 0.0]], rtol=0, atol=0)
    assert_allclose(x.grad, [[0.0, 0.0], [5e39, 0.0]], rtol=1e-15, atol=0)

def test_backward_preserves_gradient_precision_across_devices(device, assert_allclose):
    pytest.importorskip("cupy")

    target = "cuda" if device == "cpu" else "cpu"
    x = dt.Tensor([0.25], dtype=np.float64, device=device, requires_grad=True)
    intermediate = x.square(dtype=np.float16)
    moved = intermediate.to_device(target, clear_prev=False)
    weight = dt.Tensor([1e40], dtype=np.float64, device=target)

    (moved * weight).backward()

    assert intermediate.grad.dtype == np.float16
    assert moved.grad.dtype == np.float16
    assert_allclose(x.grad, [5e39], rtol=1e-15, atol=0)

def test_same_dtype_copy_preserves_gradient_precision(device, assert_allclose):
    x = dt.Tensor([-0.25, 0.25], dtype=np.float64, device=device, requires_grad=True)
    intermediate = dt.multiply(x, 2, dtype=np.float16)
    dt.relu(intermediate, in_place=True)
    copied = intermediate.copy(copy_prev=True, copy_requires_grad=True)
    weight = dt.Tensor(100_000.0, dtype=np.float64, device=device)

    with np.errstate(over="ignore"):
        (copied * weight).backward()

    assert intermediate.grad is None
    assert copied.grad.dtype == np.float16
    assert_allclose(copied.grad, [0.0, np.inf], rtol=0, atol=0)
    assert_allclose(x.grad, [0.0, 200_000.0], rtol=0, atol=0)

def test_failed_backward_restores_previous_gradients(
    device, assert_allclose
):
    x = dt.Tensor([0.25], dtype=np.float64, device=device, requires_grad=True)
    intermediate = x.square(dtype=np.float16)
    weight = dt.Tensor([100_000.0], dtype=np.float64, device=device)
    wide_branch = intermediate * weight
    other = dt.Tensor([2.0], dtype=np.float64, device=device, requires_grad=True)
    coefficient = dt.Tensor([3.0], dtype=np.float64, device=device)
    result = other * coefficient + wide_branch

    with np.errstate(over="ignore"):
        result.backward()

    tensors = (result, wide_branch, intermediate, x, other)
    gradients = [(tensor.grad, tensor.grad.copy()) for tensor in tensors]
    coefficient[0] = 8.0
    seed = dt.Tensor([2.0], dtype=np.float64, device=device)

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        result.backward(seed)

    for tensor, (buffer, values) in zip(tensors, gradients):
        assert tensor.grad is buffer
        assert_allclose(tensor.grad, values, rtol=0, atol=0)
