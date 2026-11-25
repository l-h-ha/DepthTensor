import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize(
    ("operation", "args"),
    [
        pytest.param(dt.add, (3.0,), id="add"),
        pytest.param(dt.subtract, (3.0,), id="subtract"),
        pytest.param(dt.multiply, (3.0,), id="multiply"),
        pytest.param(dt.divide, (3.0,), id="divide"),
        pytest.param(dt.power, (3.0,), id="power"),
        pytest.param(dt.matmul, ([3.0],), id="matmul"),
        pytest.param(dt.negative, (), id="negative"),
        pytest.param(dt.sign, (), id="sign"),
        pytest.param(dt.abs, (), id="abs"),
        pytest.param(dt.exp, (), id="exp"),
        pytest.param(dt.sqrt, (), id="sqrt"),
        pytest.param(dt.log, (), id="log"),
        pytest.param(dt.square, (), id="square"),
    ],
)
def test_explicit_tracking_creates_leaf(device, assert_allclose, operation, args):
    x = dt.Tensor([2.0], device=device)
    result = operation(x, *args, requires_grad=True)

    assert not operation(x, *args).requires_grad
    assert result.requires_grad
    assert result.prev == ()
    assert result.grad_fn is None

    x[0] = 4.0
    x.toggle_grad(True)
    result.sum().backward()

    assert_allclose(result.grad, np.ones(result.shape))
    assert x.grad is None

def test_explicit_tracking_with_list_operands(device, assert_allclose):
    result = dt.add([1.0], [2.0], device=device, requires_grad=True)
    result.backward(dt.Tensor([4.0], device=device))

    assert result.grad_fn is None
    assert_allclose(result, [3.0])
    assert_allclose(result.grad, [4.0])

@pytest.mark.parametrize(
    ("operation", "args", "shape", "expected"),
    [
        pytest.param(dt.add, (2.0,), (1,), [4.0], id="binary"),
        pytest.param(dt.square, (), (1,), [4.0], id="unary"),
        pytest.param(dt.mean, (), (), 2.0, id="mean"),
    ],
)
def test_explicit_tracking_rejects_in_place_before_writes(
    device, assert_allclose, operation, args, shape, expected,
):
    x = dt.Tensor([2.0], device=device)
    out = dt.zeros(shape, device=device).data
    original_data = x.data

    with pytest.raises(RuntimeError, match="In-place operations are forbidden on differentiable tensors"):
        operation(x, *args, out=out, in_place=True, requires_grad=True)

    assert x.data is original_data
    assert x._version == 0
    assert_allclose(x, [2.0])
    assert_allclose(out, np.zeros(shape))

    result = operation(x, *args, out=out, in_place=True, requires_grad=False)
    assert result is x
    assert not result.requires_grad
    assert_allclose(result, expected)

def test_explicit_tracking_validates_integer_compute_and_boolean_output(device, assert_allclose):
    x = dt.Tensor([2.0], device=device)
    float_out = dt.zeros((1,), device=device).data
    bool_out = dt.zeros((1,), device=device, dtype=np.bool_).data
    error = "Gradients require a floating-point dtype"

    with pytest.raises(RuntimeError, match=error):
        dt.add(x, 1.0, out=float_out, dtype=np.int64, casting="unsafe", requires_grad=True)

    with pytest.raises(RuntimeError, match=error):
        dt.square(x, out=bool_out, casting="unsafe", requires_grad=True)

    assert_allclose(float_out, [0.0])
    assert_allclose(bool_out, [False])
