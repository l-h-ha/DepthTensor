import pytest
import numpy as np
import depthtensor as dt

from depthtensor._core.exceptions import GradientComputationError

def test_tensor_initialization(device, assert_allclose):
    data_list = [[1.0, 2.0], [3.0, 4.0]]
    t1 = dt.Tensor(data_list, device=device)
    assert t1.device == device
    assert t1.requires_grad is False
    assert_allclose(t1, data_list)

    data_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t2 = dt.Tensor(data_np, device=device, requires_grad=True, name="test_np")
    assert t2.device == device
    assert t2.requires_grad is True
    assert t2.name == "test_np"
    assert_allclose(t2, data_np)

    t3 = dt.Tensor(t2)
    assert t3.device == device
    assert_allclose(t3, t2)

@pytest.mark.parametrize("target_device", ["", "gpu"])
def test_invalid_device_is_rejected(target_device):
    source = dt.Tensor([1.0])

    for operation in (
        lambda: dt.Tensor([1.0], device=target_device),
        lambda: dt.add(source, 1.0, device=target_device),
        lambda: source.to_device(target_device, in_place=True),
        lambda: dt.zeros((1,), device=target_device),
        lambda: dt.rand(1, device=target_device),
    ):
        with pytest.raises(ValueError, match="device must be"):
            operation()

    assert source.device == "cpu"

def test_builtin_abs_tracks_gradients(device, assert_allclose):
    source = dt.Tensor([-2.0, 0.0, 3.0], device=device, requires_grad=True)

    result = abs(source)
    result.sum().backward()

    assert_allclose(result, [2.0, 0.0, 3.0])
    assert_allclose(source.grad, [-1.0, 0.0, 1.0])

def test_tensor_properties(device):
    t = dt.Tensor([[[1, 2], [3, 4]]], device=device)

    assert t.shape == (1, 2, 2)
    assert t.ndim == 3
    assert t.size == 4

    assert t.is_cpu() == (device == "cpu")
    assert t.is_cuda() == (device == "cuda")
    assert t.is_device(device) is True

def test_tensor_to_device(device):
    t = dt.Tensor([1, 2, 3], device=device)

    t_moved = t.to_device(device, in_place=False)
    assert t_moved is not t
    assert t_moved.device == device

    t_inplace = t.to_device(device, in_place=True)
    assert t_inplace is t

def test_tensor_gradients_api(device, assert_allclose):
    t = dt.Tensor([1.0, 2.0], device=device)

    with pytest.raises(RuntimeError):
        t.zero_grad()

    t.toggle_grad(True)
    assert t.requires_grad is True

    t.zero_grad()
    assert t.grad is not None
    assert_allclose(t.grad, [0.0, 0.0])

    t.one_grad()
    assert_allclose(t.grad, [1.0, 1.0])

    t.toggle_grad(False)
    assert t.requires_grad is False
    assert t.grad is None
    assert t.grad_fn is None

def test_tensor_copy(device, assert_allclose):
    t1 = dt.Tensor([1.0, 2.0], device=device, requires_grad=True)
    t1.one_grad()

    t_default = t1.copy()
    assert t_default.requires_grad is False
    assert t_default.grad is None

    t_deep = t1.copy(copy_requires_grad=True, copy_grad=True)
    assert t_deep.requires_grad is True
    assert t_deep.grad is not None
    assert_allclose(t_deep, t1)
    assert_allclose(t_deep.grad, t1.grad)

def test_tensor_indexing(device, assert_allclose):
    data = [[1.0, 2.0], [3.0, 4.0]]
    t = dt.Tensor(data, device=device)

    row = t[0]
    assert_allclose(row, [1.0, 2.0])

    t[0, 0] = 99.0
    assert_allclose(t[0, 0], 99.0)

def test_tensor_transpose(device, assert_allclose):
    data = [[1.0, 2.0], [3.0, 4.0]]
    t = dt.Tensor(data, device=device, requires_grad=True)

    t_trans = t.transpose(axes=(1, 0))

    assert t_trans.shape == (2, 2)
    assert_allclose(t_trans, [[1.0, 3.0], [2.0, 4.0]])

    loss = t_trans.sum()
    loss.backward()

    assert t.grad is not None
    assert_allclose(t.grad, [[1.0, 1.0], [1.0, 1.0]])

def test_tensor_contiguous(device, assert_allclose):
    t = dt.Tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)

    assert t.is_contiguous() is True

    t_trans = t.transpose(axes=(1, 0))
    assert t_trans.is_contiguous() is False

    t_contig = t_trans.contiguous()
    assert t_contig.is_contiguous() is True
    assert_allclose(t_contig, [[1.0, 3.0], [2.0, 4.0]])

    loss = t_contig.sum()
    loss.backward()

    assert t.grad is not None
    assert_allclose(t.grad, [[1.0, 1.0], [1.0, 1.0]])

def test_tensor_item(device):
    t = dt.Tensor([42.0], device=device)
    val = t.item()

    assert isinstance(val, float)
    assert val == 42.0

def test_tensor_item_indices(device):
    t = dt.Tensor([[1, 2, 3], [4, 5, 6]], device=device).transpose()

    assert t.item(1) == 4
    assert t.item(-1) == 6
    assert t.item((1, 0)) == 2
    assert t.item(0, 1) == 4

    with pytest.raises(IndexError):
        t.item(6)

    with pytest.raises(IndexError):
        t.item((0, 2))

def test_tensor_iteration(device, assert_allclose):
    data = [[1.0, 2.0], [3.0, 4.0]]
    t = dt.Tensor(data, device=device)

    rows = list(t)

    assert len(rows) == 2
    assert_allclose(rows[0], [1.0, 2.0])
    assert_allclose(rows[1], [3.0, 4.0])

def test_tensor_repr_and_name(device):
    t = dt.Tensor([1, 2], device=device).set_name("bias")

    assert t.name == "bias"
    rep = repr(t)
    assert "Tensor" in rep
    assert "device=" in rep
    assert "name=bias" in rep
    assert "req_grad=False" in rep

def test_inplace_device_change_rejects_tracked_tensor(device):
    import importlib.util

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is unavailable.")

    t = dt.Tensor([1.0, 2.0], device="cpu", requires_grad=True)

    with pytest.raises(RuntimeError, match="forbidden on differentiable tensors"):
        t.to_device("cuda", in_place=True)

def test_zero_and_one_grad_require_tracking(device):
    t = dt.Tensor([1.0, 2.0], device=device, requires_grad=False)

    with pytest.raises(RuntimeError, match="undifferentiable tensor"):
        t.zero_grad()

    with pytest.raises(RuntimeError, match="undifferentiable tensor"):
        t.one_grad()

def test_backward_rejects_wrong_gradient_shape_and_device(device):
    t = dt.Tensor([1.0, 2.0], device=device, requires_grad=True)
    z = t * 2

    bad_shape_grad = dt.Tensor([1.0, 1.0, 1.0], device=device)

    with pytest.raises(RuntimeError, match="Gradient shape must match the tensor"):
        z.backward(grad=bad_shape_grad)

    import importlib.util

    if importlib.util.find_spec("cupy") is not None:
        target_device = "cuda" if device == "cpu" else "cpu"
        bad_device_grad = dt.Tensor([1.0, 1.0], device=target_device)

        with pytest.raises(RuntimeError, match="Gradient device must match the tensor"):
            z.backward(grad=bad_device_grad)

def test_backward_requires_grad_fn(device):
    t1 = dt.Tensor([1.0], device=device, requires_grad=True)
    t2 = dt.Tensor([2.0], device=device, requires_grad=True)

    z = t1 + t2

    z.grad_fn = None

    with pytest.raises(GradientComputationError, match="Missing backward function"):
        z.backward()

def test_enabling_grad_rejects_wrong_grad_shape(device):
    t = dt.Tensor([1.0, 2.0], device=device, requires_grad=True)
    t.one_grad()

    t.toggle_grad(False)

    t.grad = dt.utils.to_tensordata([1.0, 2.0, 3.0], device)

    with pytest.raises(
        RuntimeError, match="Gradient mismatch"
    ):
        t.toggle_grad(True)

def test_tensor_hash_and_dtype(device):
    t1 = dt.Tensor([1.0, 2.0], device=device, dtype=np.float32)
    t2 = dt.Tensor([1.0, 2.0], device=device)

    assert t1.dtype == np.float32

    assert hash(t1) == id(t1)

    tensor_set = {t1, t2}
    assert len(tensor_set) == 2

@pytest.mark.parametrize(
    "test_dtype, is_differentiable",
    [
        (np.float16, True),
        (np.float32, True),
        (np.float64, True),
        (np.int8, False),
        (np.int32, False),
        (np.int64, False),
        (np.uint8, False),
        (np.uint16, False),
        (bool, False),
        (np.bool_, False),
    ],
)
def test_tensor_requires_float_dtype_for_grad(device, test_dtype, is_differentiable):
    data = [1, 0, 1]

    t_no_grad = dt.Tensor(data, device=device, dtype=test_dtype, requires_grad=False)
    assert t_no_grad.dtype == test_dtype
    assert t_no_grad.requires_grad is False

    if is_differentiable:
        t_grad = dt.Tensor(data, device=device, dtype=test_dtype, requires_grad=True)
        assert t_grad.requires_grad is True
        assert t_grad.dtype == test_dtype
    else:
        with pytest.raises(
            RuntimeError, match="Gradients require a floating-point dtype"
        ):
            dt.Tensor(data, device=device, dtype=test_dtype, requires_grad=True)
