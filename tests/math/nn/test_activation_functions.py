import pytest
import numpy as np
import depthtensor as dt
from depthtensor._core.nn import functional as F
from depthtensor._core.nn._backend import relu as backend_relu

def test_relu_forward(device, assert_allclose):
    x_data = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, -3.0]], dtype=np.float32)
    x = dt.Tensor(x_data, device=device, requires_grad=True)

    y = F.relu(x)

    expected = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]
    assert_allclose(y, expected)
    assert len(y.prev) == 1

def test_relu_backward(device, assert_allclose):
    x = dt.Tensor([-5.0, 0.0, 5.0], device=device, requires_grad=True)
    y = F.relu(x)

    grad_seed = dt.Tensor([10.0, 10.0, 10.0], device=device)
    y.backward(grad=grad_seed)

    expected_grad = [0.0, 0.0, 10.0]

    assert_allclose(x.grad, expected_grad)

def test_relu_preserves_float64_range(device, assert_allclose):
    x = dt.Tensor([-1.0, 1e100], device=device, dtype=np.float64)
    y = F.relu(x)

    assert y.dtype == np.float64
    assert_allclose(y, [0.0, 1e100])

def test_relu_inplace(device, assert_allclose):
    x = dt.Tensor([-2.0, 3.0], device=device, requires_grad=False)
    initial_version = x._version

    F.relu(x, in_place=True)

    assert_allclose(x, [0.0, 3.0])
    assert x._version == initial_version + 1

def test_relu_inplace_backward(device, assert_allclose):
    x = dt.Tensor([-1.0, 2.0], device=device, requires_grad=True, dtype=np.float32)

    assert F.relu(x, in_place=True) is x

    loss = x.sum()
    loss.backward()
    assert_allclose(x.grad, [0.0, 1.0])

    loss.backward()
    assert_allclose(x.grad, [0.0, 2.0])

def test_relu_inplace_preserves_previous_backward(device, assert_allclose):
    x = dt.Tensor(
        [[-3.0, -2.0], [1.00000001, 0.5]],
        device=device,
        requires_grad=True,
        dtype=np.float64,
    )
    maxima = x.max(axis=1)

    assert F.relu(maxima, in_place=True) is maxima

    maxima.sum().backward()

    assert_allclose(maxima, [0.0, 1.00000001])
    assert_allclose(x.grad, [[0.0, 0.0], [1.0, 0.0]])

def test_repeated_inplace_relu_preserves_backward_without_recursion(device, assert_allclose):
    x = dt.Tensor([-2.0, 3.0], device=device, requires_grad=True)
    result = x * 2.0

    for _ in range(1200):
        F.relu(result, in_place=True)

    loss = result.sum()
    loss.backward()
    loss.backward()

    assert_allclose(result.grad, [0.0, 2.0])
    assert_allclose(x.grad, [0.0, 4.0])

def test_relu_out_tracks_writes_and_protects_tracked_buffers(device, assert_allclose):
    x = dt.Tensor([-2.0, 3.0], device=device, requires_grad=True)
    tracked_out = dt.Tensor([2.0, 4.0], device=device, requires_grad=True)
    original_loss = tracked_out.square().sum()

    with pytest.raises(RuntimeError, match="out buffer"):
        backend_relu(x, out=tracked_out.data)

    original_loss.backward()
    assert_allclose(tracked_out.grad, [4.0, 8.0])

    out = dt.Tensor([1.0, 2.0], device=device)
    previous_loss = (x * out).sum()
    result = backend_relu(x, out=out.data)

    assert result.data is out.data
    assert_allclose(out, [0.0, 3.0])

    with pytest.raises(RuntimeError, match="modified"):
        previous_loss.backward()

    result.sum().backward()
    assert_allclose(x.grad, [0.0, 1.0])

@pytest.mark.parametrize("dtype, positive", [(np.int64, 2**40 + 1), (np.float16, 1.5)])
@pytest.mark.parametrize("in_place", [False, True])
def test_relu_preserves_input_dtype_and_exact_values(device, dtype, positive, in_place):
    x = dt.Tensor([-2, positive], device=device, dtype=dtype)

    result = F.relu(x, in_place=in_place)

    assert result.dtype == dtype
    assert result[0].item() == 0
    assert result[1].item() == positive

@pytest.mark.parametrize("in_place", [False, True])
def test_relu_preserves_boolean_dtype(device, in_place):
    x = dt.Tensor([False, True], device=device)

    result = F.relu(x, in_place=in_place)

    assert result.dtype == np.bool_
    assert [result[0].item(), result[1].item()] == [False, True]
    assert (result is x) == in_place

@pytest.mark.parametrize("source", ["cpu", "cuda"])
def test_inplace_relu_device_move_updates_untracked_storage(source, assert_allclose):
    cp = pytest.importorskip("cupy")

    destination = "cuda" if source == "cpu" else "cpu"
    x = dt.Tensor([-3.0, 5.0], device=source)

    assert backend_relu(x, device=destination, in_place=True) is x
    assert x.device == destination
    assert isinstance(x.data, np.ndarray if destination == "cpu" else cp.ndarray)
    assert_allclose(x, [0.0, 5.0])
    assert x._version == 1

@pytest.mark.parametrize("source", ["cpu", "cuda"])
def test_inplace_relu_device_move_rejects_tracked_tensors(source, assert_allclose):
    pytest.importorskip("cupy")

    destination = "cuda" if source == "cpu" else "cpu"
    x = dt.Tensor([-3.0, 5.0], device=source, requires_grad=True)
    out = dt.Tensor([4.0, 6.0], device=destination)

    with pytest.raises(RuntimeError, match="In-place device changes are forbidden"):
        backend_relu(x, out=out.data, device=destination, in_place=True)

    assert x.device == source
    assert x._version == out._version == 0
    assert_allclose(x, [-3.0, 5.0])
    assert_allclose(out, [4.0, 6.0])

def test_cpu_inplace_relu_invalidates_independent_storage_alias():
    storage = np.array([-2.0, 3.0])
    tracked = dt.Tensor(storage, requires_grad=True)
    alias = dt.Tensor(storage)
    loss = tracked.square().sum()

    F.relu(alias, in_place=True)

    np.testing.assert_array_equal(tracked.data, [0.0, 3.0])

    with pytest.raises(RuntimeError, match="modified"):
        loss.backward()

def test_cpu_inplace_relu_rejects_overwriting_differentiable_result(assert_allclose):
    x = dt.Tensor([-2.0, 4.0], requires_grad=True)
    result = x * 3.0
    alias = dt.Tensor(result)
    version = result._version

    with pytest.raises(RuntimeError, match="In-place ReLU"):
        F.relu(alias, in_place=True)

    assert_allclose(result, [-6.0, 12.0])
    assert result._version == alias._version == version

    result.sum().backward()
    assert_allclose(x.grad, [3.0, 3.0])

@pytest.mark.parametrize("in_place", [False, True])
def test_relu_masks_nonfinite_gradients_on_inactive_branch(device, in_place, assert_allclose):
    x = dt.Tensor([-2.0, 3.0], device=device, requires_grad=True)
    seed = dt.Tensor([np.inf, 1.0], device=device)

    F.relu(x, in_place=in_place).backward(seed)

    assert_allclose(x.grad, [0.0, 1.0])

def test_tracked_relu_rejects_integer_out_before_writing(device, assert_allclose):
    x = dt.Tensor([-2.0, 3.0], device=device, requires_grad=True)
    out = dt.Tensor([4, 5], device=device, dtype=np.int32)

    with pytest.raises(RuntimeError, match="Gradients require a floating-point dtype"):
        backend_relu(x, out=out.data)

    assert_allclose(out, [4, 5])
    assert out._version == 0
