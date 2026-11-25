import numpy as np
import pytest
import depthtensor as dt
from depthtensor._core.nn._backend.activation_functions import relu

@pytest.fixture(params=[
    pytest.param((dt.add, (1.0,), {}), id="add"),
    pytest.param((dt.clip, (0.0, 3.0), {}), id="clip"),
    pytest.param((dt.greater, (1.0,), {}), id="greater"),
    pytest.param((dt.mean, (), {"axis": ()}), id="mean"),
    pytest.param((relu, (), {}), id="relu"),
])
def output_operation(request):
    return request.param

@pytest.mark.parametrize("method, kwargs", [
    pytest.param("sum", {}, id="sum"),
    pytest.param("sum", {"initial": 1.0}, id="sum-initial"),
    pytest.param("mean", {}, id="mean"),
    pytest.param("max", {}, id="max"),
])
def test_reduction_accepts_one_output_buffer_tuple(device, assert_allclose, method, kwargs):
    values = np.arange(1.0, 7.0).reshape(2, 3)
    source = dt.Tensor(values, device=device)
    destination = dt.zeros((2, 1), dtype=np.float32, device=device)
    version = destination._version

    result = getattr(source, method)(axis=-1, keepdims=True, out=(destination.data,), **kwargs)

    assert result.data is destination.data
    assert result._version == destination._version == version + 1
    assert_allclose(result, getattr(np, method)(values, axis=-1, keepdims=True, **kwargs))

@pytest.mark.parametrize("method, expected", [("sum", 1.0), ("mean", 1.0 / 3.0)])
def test_cpu_reduction_widens_accumulation_to_output_dtype(method, expected):
    source = dt.Tensor([16777216.0, 1.0, -16777216.0], dtype=np.float32, device="cpu")
    destination = dt.zeros((), dtype=np.float64, device="cpu")

    result = getattr(source, method)(out=destination.data)

    assert result.data is destination.data
    assert result.item() == expected

def test_max_casts_integer_result_without_losing_precision(device):
    source = dt.Tensor([0, 2**63 - 1], dtype=np.uint64, device=device)
    destination = dt.zeros((), dtype=np.int8, device=device)

    result = source.max(out=destination.data)

    assert result.data is destination.data
    assert result.item() == -1

@pytest.mark.parametrize("method, kwargs", [
    pytest.param("sum", {"initial": 1.0}, id="sum"),
    pytest.param("mean", {}, id="mean"),
    pytest.param("max", {}, id="max"),
])
@pytest.mark.parametrize("keepdims, wrong_shape", [(False, (2, 2)), (True, (2, 3))])
def test_reduction_rejects_broadcastable_output_before_writing(
    device, assert_allclose, method, kwargs, keepdims, wrong_shape
):
    source = dt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    untouched = np.full(wrong_shape, -9.0)
    destination = dt.Tensor(untouched.copy(), device=device)
    version = destination._version

    with pytest.raises(ValueError):
        getattr(source, method)(axis=-1, keepdims=keepdims, out=destination.data, **kwargs)

    assert destination._version == version
    assert_allclose(destination, untouched)

@pytest.mark.parametrize("method, destination_dtype, values", [
    pytest.param("sum", np.float64, [[1.0, 2.0], [1e308, 1e308]], id="accumulation"),
    pytest.param("max", np.float16, [[1.0, 2.0], [3.0, 1e5]], id="output-cast"),
])
def test_failed_cpu_reduction_leaves_output_unchanged(
    method, destination_dtype, values, assert_allclose
):
    source = dt.Tensor(values, dtype=np.float64, device="cpu")
    destination = dt.Tensor([-9.0, -9.0], dtype=destination_dtype, device="cpu")
    version = destination._version

    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="overflow"):
        getattr(source, method)(axis=1, out=destination.data)

    assert destination._version == version
    assert_allclose(destination, [-9.0, -9.0])

def test_failed_cpu_divide_leaves_output_unchanged(assert_allclose):
    numerator = dt.Tensor([2.0, 1.0], device="cpu")
    destination = dt.Tensor([-9.0, -9.0], device="cpu")
    version = destination._version

    with np.errstate(divide="raise"), pytest.raises(FloatingPointError, match="divide by zero"):
        dt.divide(numerator, [2.0, 0.0], out=destination.data)

    assert destination._version == version
    assert_allclose(destination, [-9.0, -9.0])

def test_failed_cpu_mean_leaves_output_unchanged():
    source = dt.Tensor([1.0, 2.0], device="cpu")
    destination = dt.Tensor(-9.0, device="cpu")
    version = destination._version

    with np.errstate(invalid="raise"), pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        with pytest.raises(FloatingPointError):
            source.mean(where=False, out=destination.data)

    assert destination._version == version
    assert destination.item() == -9.0

def test_masked_pointwise_broadcasts_to_output_shape(device, assert_allclose):
    source = dt.Tensor([1.0, 7.0], device=device, requires_grad=True)
    destination = dt.Tensor(np.full((2, 2), -9.0), device=device)
    version = destination._version

    result = dt.add(source, 2.0, out=destination.data, where=[True, False])
    result.sum().backward()

    assert result.data is destination.data
    assert result._version == destination._version == version + 1
    assert_allclose(result, [[3.0, -9.0], [3.0, -9.0]])
    assert_allclose(source.grad, [2.0, 0.0])

def test_masked_write_uses_original_mask_when_it_overlaps_output(device, assert_allclose):
    source = dt.Tensor([0.0, 1.0], device=device)
    destination = dt.Tensor([True, True], device=device)

    result = dt.greater(source, 2.0, out=destination.data, where=destination.data[:1])

    assert result.data is destination.data
    assert_allclose(destination, [False, False])

def test_output_requires_array(output_operation, assert_allclose):
    operation, args, kwargs = output_operation
    source = dt.Tensor([1.0, 2.0], device="cpu")
    destination = dt.Tensor([-9.0, -9.0], device="cpu")
    version = destination._version

    with pytest.raises(TypeError):
        operation(source, *args, out=destination, **kwargs)

    with pytest.raises(ValueError):
        operation(source, *args, out=(destination.data, destination.data), **kwargs)

    assert destination._version == version
    assert_allclose(destination, [-9.0, -9.0])

def test_read_only_cpu_output_rejected_before_writing(output_operation, assert_allclose):
    operation, args, kwargs = output_operation
    source = dt.Tensor([1.0, 2.0], device="cpu")
    destination = dt.Tensor([-9.0, -9.0], device="cpu")
    destination.data.flags.writeable = False
    version = destination._version

    with pytest.raises(ValueError, match="out buffer must be writable"):
        operation(source, *args, out=destination.data, **kwargs)

    assert destination._version == version
    assert_allclose(destination, [-9.0, -9.0])

@pytest.mark.parametrize("source_device, out_device", [("cpu", "cuda"), ("cuda", "cpu")])
def test_output_must_match_operation_device(source_device, out_device, assert_allclose):
    pytest.importorskip("cupy")

    source = dt.Tensor([1.0, 2.0], device=source_device)
    destination = dt.Tensor([-9.0, -9.0], device=out_device)
    version = destination._version

    with pytest.raises(ValueError, match=f"out buffer must be on the {source_device} device"):
        dt.add(source, 1.0, out=destination.data)

    assert destination._version == version
    assert_allclose(destination, [-9.0, -9.0])
