import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize("mask", [True, [True, False]], ids=["unmasked", "masked"])
def test_clip_infers_compute_dtype_with_unsafe_cast(
    device, mask, assert_allclose
):
    x = dt.Tensor([0.25, 1.25], dtype=np.float64, device=device)
    out = dt.Tensor([-9, -9], dtype=np.int64, device=device).data

    result = dt.clip(x, 0.0, 2.0, out=out, where=mask, casting="unsafe")

    assert result.data is out
    assert_allclose(result, [0, 1 if mask is True else -9])

def test_clip_no_casting_rejects_narrower_output_before_writing(device, assert_allclose):
    x = dt.Tensor([0.25, 1.25], dtype=np.float64, device=device)
    out = dt.Tensor([-9.0, -9.0], dtype=np.float32, device=device)

    with pytest.raises(TypeError, match="casting"):
        dt.clip(x, 0.0, 2.0, out=out.data, casting="no")

    assert_allclose(out, [-9.0, -9.0])

def test_clip_no_casting_rejects_mixed_operand_dtypes(device):
    x = dt.Tensor([0.25, 1.25], dtype=np.float64, device=device)
    lower = dt.Tensor([0.0], dtype=np.float32, device=device)

    with pytest.raises(TypeError, match="casting"):
        dt.clip(x, lower, 2.0, casting="no")

def test_masked_add_preserves_out_and_uses_forward_mask(device, assert_allclose):
    left = dt.Tensor([[1.0], [2.0]], device=device, requires_grad=True)
    right = dt.Tensor([[3.0, 4.0, 5.0]], device=device, requires_grad=True)
    out = dt.Tensor(np.full((2, 3), 99.0), device=device).data
    mask = dt.Tensor([True, False, True], device=device).data

    result = dt.add(left, right, out=out, where=mask, requires_grad=True)
    assert result.data is out
    assert_allclose(out, [[4.0, 99.0, 6.0], [5.0, 99.0, 7.0]])

    mask[...] = False
    result.sum().backward()

    assert_allclose(left.grad, [[2.0], [2.0]])
    assert_allclose(right.grad, [[2.0, 0.0, 2.0]])

def test_masked_add_uses_output_casting(device, assert_allclose):
    x = dt.Tensor([1.25, 2.25], device=device)
    out = dt.Tensor([99, 99], dtype=np.int32, device=device).data

    result = dt.add(x, 3.0, out=out, where=[True, False], casting="unsafe")

    assert result.data is out
    assert result.dtype == np.int32
    assert_allclose(out, [4, 99])

@pytest.mark.parametrize(
    "operation, args, expected",
    [(dt.add, (2,), 3.0), (dt.clip, (0, 5), 1.0)],
    ids=["add", "clip"],
)
def test_masked_pointwise_accepts_a_single_output_tuple(
    device, assert_allclose, operation, args, expected
):
    x = dt.Tensor([1.0, 7.0], device=device, requires_grad=True)
    out = dt.Tensor([-9.0, -9.0], device=device)

    result = operation(x, *args, out=(out.data,), where=[True, False])
    result.sum().backward()

    assert result.data is out.data
    assert result._version == out._version == 1
    assert_allclose(out, [expected, -9.0])
    assert_allclose(x.grad, [1.0, 0.0])

def test_masked_log_skips_excluded_derivatives(device, assert_allclose):
    x = dt.Tensor([2.0, 0.0], device=device, requires_grad=True)
    out = dt.Tensor([99.0, 99.0], device=device).data

    with np.errstate(divide="raise", invalid="raise"):
        result = dt.log(x, out=out, where=[True, False], requires_grad=True)
        result.sum().backward()

    assert_allclose(out, [np.log(2.0), 99.0])
    assert_allclose(x.grad, [0.5, 0.0])

def test_masked_unary_backward_reduces_broadcast_gradient(device, assert_allclose):
    x = dt.Tensor([3.0], device=device, requires_grad=True)
    out = dt.Tensor([0.0, 99.0, 0.0], device=device).data

    dt.square(x, out=out, where=[True, False, True], requires_grad=True).sum().backward()

    assert_allclose(out, [9.0, 99.0, 9.0])
    assert_allclose(x.grad, [12.0])

def test_masked_clip_routes_gradients_to_selected_bounds(device, assert_allclose):
    x = dt.Tensor([-1.0, 4.0, 8.0], device=device, requires_grad=True)
    lower = dt.Tensor(0.0, device=device, requires_grad=True)
    upper = dt.Tensor(5.0, device=device, requires_grad=True)
    out = dt.Tensor([99.0, 99.0, 99.0], device=device).data

    dt.clip(x, lower, upper, out=out, where=[True, False, True]).sum().backward()

    assert_allclose(out, [0.0, 99.0, 5.0])
    assert_allclose(x.grad, [0.0, 0.0, 0.0])
    assert_allclose(lower.grad, 1.0)
    assert_allclose(upper.grad, 1.0)

def test_binary_device_override_tracks_gradients_on_input_devices(device, assert_allclose):
    cp = pytest.importorskip("cupy")

    left = dt.Tensor([1.0, 2.0], device="cpu", requires_grad=True)
    right = dt.Tensor([3.0, 4.0], device="cuda", requires_grad=True)

    result = dt.multiply(left, right, device=device, requires_grad=True)
    assert result.device == device
    assert isinstance(result.data, np.ndarray if device == "cpu" else cp.ndarray)
    assert_allclose(result, [3.0, 8.0])

    result.sum().backward()

    assert isinstance(left.grad, np.ndarray)
    assert isinstance(right.grad, cp.ndarray)
    assert_allclose(left.grad, [3.0, 4.0])
    assert_allclose(right.grad, [1.0, 2.0])

def test_in_place_binary_device_override_updates_metadata(device, assert_allclose):
    cp = pytest.importorskip("cupy")

    source_device = "cuda" if device == "cpu" else "cpu"
    x = dt.Tensor([1.0, 2.0], device=source_device)

    result = dt.add(x, 3.0, device=device, in_place=True)

    assert result is x
    assert result.device == device
    assert isinstance(result.data, np.ndarray if device == "cpu" else cp.ndarray)
    assert_allclose(result, [4.0, 5.0])

def test_python_scalar_left_operands_use_tensor_device(device, assert_allclose):
    x = dt.Tensor([1.0, 4.0], device=device)

    for result, expected in (
        (2 + x, [3.0, 6.0]),
        (2 - x, [1.0, -2.0]),
        (2 / x, [2.0, 0.5]),
    ):
        assert result.device == device
        assert_allclose(result, expected)

def test_cpu_reduction_scalar_can_be_used_on_cuda(assert_allclose):
    cp = pytest.importorskip("cupy")

    source = dt.Tensor([2.0, 3.0], dtype=np.float32, device="cpu", requires_grad=True)
    scalar = source.sum()
    vector = dt.Tensor([1.0, 2.0], dtype=np.float32, device="cuda")

    result = dt.add(vector, scalar, device="cuda")
    assert isinstance(result.data, cp.ndarray)
    assert_allclose(result, [6.0, 7.0])

    result.sum().backward()
    assert_allclose(source.grad, [2.0, 2.0])

    empty = dt.zeros_like(scalar, device="cuda")
    assert isinstance(empty.data, cp.ndarray)
    assert empty.shape == ()
    assert empty.dtype == scalar.dtype
    assert_allclose(empty, 0.0)

    untracked_scalar = dt.Tensor([2.0, 3.0], dtype=np.float32, device="cpu").sum()
    assert untracked_scalar.to_device("cuda", in_place=True) is untracked_scalar
    assert isinstance(untracked_scalar.data, cp.ndarray)
    assert untracked_scalar.device == "cuda"
    assert untracked_scalar.shape == ()
    assert untracked_scalar.dtype == np.float32
    assert_allclose(untracked_scalar, 5.0)
