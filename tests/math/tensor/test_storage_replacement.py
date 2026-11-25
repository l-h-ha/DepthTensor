import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize(
    "replace, values, dtype, keep_grad",
    [
        pytest.param(lambda x: x.mean(in_place=True), 3.0, np.float64, False, id="shape"),
        pytest.param(
            lambda x: x.square(dtype=np.float32, in_place=True),
            [4.0, 16.0],
            np.float32,
            False,
            id="dtype",
        ),
        pytest.param(lambda x: dt.add(x, 1, in_place=True), [3.0, 5.0], np.float64, True, id="compatible"),
    ],
)
def test_inplace_replacement_keeps_only_compatible_gradients(
    device, assert_allclose, replace, values, dtype, keep_grad
):
    source = dt.Tensor([2.0, 4.0], dtype=np.float64, device=device, requires_grad=True)
    source.one_grad()
    target = source.copy(copy_grad=True)
    alias = dt.Tensor(target)
    previous_grad = target.grad
    version = target._version

    assert replace(target) is target
    assert target.device == device
    assert target.dtype == dtype
    assert target._version == version + 1
    assert alias._version == version
    assert_allclose(target, values)
    assert_allclose(alias, [2.0, 4.0])
    assert target.grad is (previous_grad if keep_grad else None)

    target.toggle_grad(True)
    target.backward()
    assert_allclose(target.grad, np.ones_like(values, dtype=dtype))

def test_inplace_transfer_clears_a_gradient_on_the_old_device(device, assert_allclose):
    pytest.importorskip("cupy")

    other = "cuda" if device == "cpu" else "cpu"
    source = dt.Tensor([2.0, 4.0], device=device, requires_grad=True)
    source.one_grad()
    target = source.copy(copy_grad=True)
    version = target._version

    assert target.to_device(other, in_place=True) is target
    assert target.device == other
    assert dt.utils.get_device(target.data) == other
    assert target._version == version + 1
    assert target.grad is None

    target.toggle_grad(True)
    target.backward()
    assert dt.utils.get_device(target.grad) == other
    assert_allclose(target.grad, [1.0, 1.0])

def test_inplace_relu_preserves_compatible_gradient_and_backward(device, assert_allclose):
    target = dt.Tensor([-2.0, 4.0], device=device, requires_grad=True)
    target.one_grad()
    previous_grad = target.grad
    version = target._version

    assert dt.relu(target, in_place=True) is target
    assert target.grad is previous_grad
    assert target._version == version + 1

    target.sum().backward()
    assert_allclose(target.grad, [1.0, 2.0])
