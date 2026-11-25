import numpy as np
import pytest

import depthtensor as dt

def test_where_backward_does_not_evaluate_inactive_sqrt_at_zero(device, assert_allclose):
    x = dt.Tensor([0.0, 4.0], device=device, requires_grad=True)
    result = dt.where([False, True], x.sqrt(), x)

    with np.errstate(divide="raise", invalid="raise"):
        result.sum().backward()

    assert_allclose(result, [0.0, 2.0])
    assert_allclose(x.grad, [1.0, 0.25])

def test_where_backward_masks_singular_division_before_unbroadcast(device, assert_allclose):
    x = dt.Tensor([4.0], device=device, requires_grad=True)
    with np.errstate(divide="ignore"):
        ratios = dt.divide(x, [0.0, 2.0])
    result = dt.where([False, True], ratios, 0.0)

    with np.errstate(divide="raise", invalid="raise"):
        result.sum().backward()

    assert_allclose(result, [0.0, 2.0])
    assert_allclose(x.grad, [0.5])

def test_active_sqrt_at_zero_preserves_infinite_gradient(device, assert_allclose):
    x = dt.Tensor([0.0, 4.0], device=device, requires_grad=True)
    with np.errstate(divide="ignore"):
        x.sqrt().sum().backward()

    assert_allclose(x.grad, [np.inf, 0.25])

@pytest.mark.parametrize(
    ("operation", "args", "out_shape", "kwargs"),
    [
        pytest.param(dt.add, (0.0,), (2,), {"casting": "unsafe"}, id="add"),
        pytest.param(dt.negative, (), (2,), {"casting": "unsafe"}, id="negative"),
        pytest.param(dt.clip, (0.0, 10.0), (2,), {"casting": "unsafe"}, id="clip"),
        pytest.param(dt.maximum, (0.0,), (2,), {"casting": "unsafe"}, id="maximum"),
        pytest.param(dt.sum, (), (), {}, id="sum"),
        pytest.param(dt.mean, (), (), {}, id="mean"),
    ],
)
def test_grad_tracking_rejects_integer_compute_dtype_before_out_write(
    device, assert_allclose, operation, args, out_shape, kwargs,
):
    x = dt.Tensor([1.25, 2.75], device=device, requires_grad=True)
    owner = dt.ones(out_shape, device=device, dtype=np.float64)

    with pytest.raises(RuntimeError, match="Gradients require a floating-point dtype"):
        operation(x, *args, out=owner.data, dtype=np.int64, **kwargs)

    assert_allclose(owner, np.ones(out_shape))
    assert owner._version == 0

def test_explicitly_detached_operation_allows_integer_compute_dtype(device, assert_allclose):
    x = dt.Tensor([1.25, 2.75], device=device, requires_grad=True)
    out = dt.zeros((2,), device=device, dtype=np.float64).data
    result = dt.add(x, 0.0, dtype=np.int64, casting="unsafe", out=out, requires_grad=False)

    assert result.data is out
    assert not result.requires_grad
    assert_allclose(result, [1.0, 2.0])

@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(dt.multiply, id="multiply"),
        pytest.param(dt.maximum, id="maximum"),
        pytest.param(lambda x, bound: dt.clip(x, bound, None), id="clip"),
        pytest.param(lambda x, other: dt.where(False, x, other, device=x.device), id="where"),
    ],
)
def test_enabling_gradients_later_does_not_add_a_parent(device, operation):
    tracked = dt.Tensor([1.0], device=device, requires_grad=True)
    constant = dt.Tensor([2.0], device=device)
    result = operation(tracked, constant).sum()

    constant.toggle_grad(True)
    result.backward()

    assert constant.grad is None

def test_failed_backward_leaves_gradients_unset(device):
    earlier = dt.Tensor([3.0], device=device, requires_grad=True)
    tracked = dt.Tensor([4.0], device=device, requires_grad=True)
    constant = dt.Tensor([2.0], device=device)
    loss = (earlier.square() + tracked * constant).sum()
    constant.toggle_grad(True)
    earlier[0] = 7.0

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

    assert constant.grad is None
    assert tracked.grad is None

@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_divide_backward_avoids_squaring_extreme_denominators(device, assert_allclose, scale):
    denominator = dt.Tensor([scale], device=device, dtype=np.float64, requires_grad=True)
    ratio = scale / denominator

    ratio.sum().backward()

    assert_allclose(ratio, [1.0])
    assert_allclose(denominator.grad, [-1.0 / scale], atol=0)

@pytest.mark.parametrize(
    ("numerator", "denominator", "seed", "expected"),
    [
        (1e308, 1e200, 1e-200, -1e-292),
        (1e-308, 1e-200, 1e200, -1e292),
    ],
)
def test_divide_backward_scales_extreme_seed_before_rounding(
    device, assert_allclose, numerator, denominator, seed, expected,
):
    divisor = dt.Tensor([denominator], device=device, dtype=np.float64, requires_grad=True)
    ratio = numerator / divisor
    upstream = dt.Tensor([seed], device=device, dtype=np.float64)
    ratio.backward(upstream)

    assert_allclose(divisor.grad, [expected], atol=0)

def test_power_base_backward_with_subnormal_base(device, assert_allclose):
    base = 1e-320
    exponent = 1e-100
    x = dt.Tensor([base], device=device, dtype=np.float64, requires_grad=True)
    result = dt.power(x, exponent)
    result.sum().backward()

    assert_allclose(result, [1.0])
    assert_allclose(x.grad, [exponent / base], atol=0)

def test_power_exponent_backward_at_zero_base(device, assert_allclose):
    exponent = dt.Tensor([0.5, 2.0, 2.0], device=device, requires_grad=True)
    dt.power([0.0, 0.0, 2.0], exponent).sum().backward()

    assert_allclose(exponent.grad, [0.0, 0.0, 4 * np.log(2)])

def test_clip_backward_with_equal_bounds(device, assert_allclose):
    x = dt.Tensor([0.5, 1.0, 1.5], device=device, requires_grad=True)
    lower = dt.Tensor(1.0, device=device, requires_grad=True)
    upper = dt.Tensor(1.0, device=device, requires_grad=True)
    result = dt.clip(x, lower, upper)
    result.sum().backward()

    assert_allclose(result, [1.0, 1.0, 1.0])
    assert_allclose(x.grad, [0.0, 0.0, 0.0])
    assert_allclose(lower.grad, 0.0)
    assert_allclose(upper.grad, 3.0)

def test_multiply_backward_uses_explicit_computation_dtype(device, assert_allclose):
    x = dt.Tensor([1.0], device=device, requires_grad=True)
    y = dt.Tensor([101.1], device=device)

    product = dt.multiply(x, y, dtype=np.float16, casting="unsafe")
    product.sum().backward()

    rounded = float(np.float16(101.1))
    assert_allclose(product, [rounded], rtol=0, atol=0)
    assert_allclose(x.grad, [rounded], rtol=0, atol=0)

def test_masked_mean_rounds_only_the_final_result_to_out_dtype(device, assert_allclose):
    x = dt.Tensor([60000.0, 60000.0, 1.0], device=device, dtype=np.float32, requires_grad=True)
    out = dt.zeros((), dtype=np.float16, device=device).data
    result = dt.mean(x, where=[True, True, False], out=out)

    assert result.data is out
    assert result.item() == 60000.0
    result.backward()
    assert_allclose(x.grad, [0.5, 0.5, 0.0])

def test_unmasked_cpu_mean_rounds_only_the_final_result_to_out_dtype(assert_allclose):
    x = dt.Tensor([60000.0] * 3, dtype=np.float32, device="cpu", requires_grad=True)
    out = np.empty((), dtype=np.float16)
    result = dt.mean(x, out=out)

    assert result.data is out
    assert result.item() == 60000.0
    result.backward()
    assert_allclose(x.grad, [1 / 3] * 3, rtol=3e-4)

def test_mean_disables_tracking_with_requires_grad_false(device):
    tracked = dt.Tensor([1.0, 3.0], device=device, requires_grad=True)
    detached = dt.mean(tracked, requires_grad=False)

    assert not detached.requires_grad
    assert detached.grad_fn is None
    assert dt.mean(tracked).requires_grad

def test_in_place_flag_without_tensor_destination_validates_out_dtype(device, assert_allclose):
    tracked = dt.Tensor([2.5], device=device, requires_grad=True)
    out = dt.ones((1,), dtype=np.int64, device=device).data

    with pytest.raises(RuntimeError, match="Gradients require a floating-point dtype"):
        dt.add([1.0], tracked, out=out, in_place=True, casting="unsafe")

    assert_allclose(out, [1])

def test_in_place_binary_requires_explicit_detach_with_tracked_right_operand(device, assert_allclose):
    target = dt.Tensor([1.0], device=device)
    tracked = dt.Tensor([2.5], device=device, requires_grad=True)
    out = dt.zeros((1,), device=device).data
    original_data = target.data
    original_version = target._version

    with pytest.raises(RuntimeError, match="In-place operations are forbidden on differentiable tensors"):
        dt.add(target, tracked, in_place=True, out=out)

    assert target.data is original_data
    assert target._version == original_version
    assert_allclose(out, [0.0])

    result = dt.add(target, tracked, in_place=True, out=out, requires_grad=False)
    assert result is target
    assert not result.requires_grad
    assert_allclose(result, [3.5])
    assert_allclose(out, [3.5])
