import numpy as np
import depthtensor as dt

def test_ones_like(device, assert_allclose):
    source = dt.Tensor([2.0, 3.0, 4.0], device=device, dtype=np.float32)
    result = dt.ones_like(source, requires_grad=True)

    assert result.device == device
    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert result.requires_grad
    assert_allclose(result, [1.0, 1.0, 1.0])

def test_zeros_like_to_cpu(device, assert_allclose):
    source = dt.Tensor([[2.0, 3.0], [4.0, 5.0]], device=device, dtype=np.float32)
    result = dt.zeros_like(source, device="cpu", requires_grad=True)

    assert result.device == "cpu"
    assert isinstance(result.data, np.ndarray)
    assert result.dtype == source.dtype
    assert result.requires_grad
    assert_allclose(result, [[0.0, 0.0], [0.0, 0.0]])
