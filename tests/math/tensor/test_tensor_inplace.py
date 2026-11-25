import pytest
import operator
import numpy as np
import depthtensor as dt

@pytest.mark.parametrize(
    "op_func, np_func",
    [
        (operator.iadd, np.add), # +=
        (operator.isub, np.subtract), # -=
        (operator.imul, np.multiply), # *=
        (operator.itruediv, np.divide), # /=
        (operator.ipow, np.power), # **=
        (operator.imatmul, np.matmul), # @=
    ],
)
def test_inplace_operations(device, assert_allclose, op_func, np_func):
    a_data = [[2.0, 3.0], [4.0, 5.0]]
    b_data = [[1.0, 2.0], [3.0, 4.0]]

    a = dt.Tensor(a_data, device=device, requires_grad=False)
    b = dt.Tensor(b_data, device=device, requires_grad=False)

    a_np = np.array(a_data)
    b_np = np.array(b_data)

    original_id = id(a)

    a = op_func(a, b)
    expected = np_func(a_np, b_np)

    assert (
        id(a) == original_id
    ), f"In-place {op_func.__name__} allocated a new Tensor object."

    assert_allclose(a, expected)
