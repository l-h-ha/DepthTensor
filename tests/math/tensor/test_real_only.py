import numpy as np
import pytest

import depthtensor as dt
from depthtensor import _ext

def _complex_storage(device):
    xp = np if device == "cpu" else pytest.importorskip("cupy")
    return xp.array([4 + 5j], dtype=np.complex64)

@pytest.mark.parametrize("dtype", [None, np.float64])
@pytest.mark.parametrize("data", [1 + 2j, np.complex64(1j), [[1, 2j]], np.array([1j])])
def test_tensor_rejects_complex_values(device, data, dtype):
    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        dt.Tensor(data, dtype=dtype, device=device)

def test_cpu_tensor_rejects_complex_values_in_object_storage():
    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        dt.Tensor(np.array([1j], dtype=object))

def test_native_relu_rejects_complex_storage():
    with pytest.raises(TypeError):
        _ext.nn.relu_forward_cpu(np.array([-1 + 2j]))

@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda x: dt.Tensor(x, dtype=np.complex64), id="tensor"),
        pytest.param(lambda x: dt.zeros_like(x, dtype="complex64"), id="zeros_like"),
        pytest.param(lambda x: dt.ones(x.shape, dtype=complex, device=x.device), id="ones"),
        pytest.param(lambda x: dt.exp(x, dtype=np.complex64), id="exp"),
        pytest.param(lambda x: dt.add(x, 2, dtype=np.complex64), id="add"),
        pytest.param(lambda x: dt.equal(x, 2, dtype=np.complex64), id="equal"),
        pytest.param(lambda x: dt.maximum(x, 2, dtype=np.complex64), id="maximum"),
        pytest.param(lambda x: dt.clip(x, 0, 3, dtype=np.complex64), id="clip"),
        pytest.param(lambda x: dt.sum(x, dtype=np.complex64), id="sum"),
        pytest.param(lambda x: dt.mean(x, dtype=np.complex64), id="mean"),
        pytest.param(lambda x: dt.uniform(device=x.device, dtype=np.complex64), id="uniform"),
    ],
)
def test_operations_reject_complex_dtypes(device, operation):
    x = dt.Tensor([2.0], device=device)

    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        operation(x)

def test_operations_reject_complex_operands_and_masks(device):
    x = dt.Tensor([3.0], device=device)
    complex_data = _complex_storage(device)
    for operation in (
        lambda: x + (2 + 1j),
        lambda: dt.multiply(x, complex_data, dtype=np.float64, casting="unsafe"),
        lambda: dt.where(complex_data, x, 0),
        lambda: dt.clip(x, complex_data, None),
    ):
        with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
            operation()

def test_complex_outputs_and_seeds_leave_existing_storage_unchanged(device, assert_allclose):
    x = dt.Tensor([3.0], device=device, requires_grad=True)
    x.one_grad()
    previous_grad = x.grad
    complex_data = _complex_storage(device)

    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        dt.add(x, 2, out=(complex_data,))

    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        x.backward(complex_data)

    assert x.grad is previous_grad
    assert_allclose(x.grad, [1.0])
    assert_allclose(complex_data, [4 + 5j])

def test_tensor_rejects_complex_storage_index_and_assignment(device, assert_allclose):
    x = dt.Tensor([3.0], device=device)
    version = x._version
    complex_data = _complex_storage(device)

    for attribute in ("data", "grad"):
        with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
            setattr(x, attribute, complex_data)

    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        x[complex_data]

    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        x[0] = complex_data[0]

    assert x._version == version
    assert x.grad is None
    assert_allclose(x, [3.0])

def test_tensor_rejects_complex_legacy_state():
    x = object.__new__(dt.Tensor)
    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        x.__setstate__({"data": np.array([1j])})

def test_tensor_applies_explicit_dtype_before_inference(device):
    unsigned = dt.Tensor([0, 2**64 - 1], dtype=np.uint64, device=device)
    signed = dt.Tensor([2**53 + 1, 0.0], dtype=np.int64, device=device)

    assert unsigned.dtype == np.uint64
    assert signed.dtype == np.int64
    assert unsigned[1].item() == 2**64 - 1
    assert signed[0].item() == 2**53 + 1
