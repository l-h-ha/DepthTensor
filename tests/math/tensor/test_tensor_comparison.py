import pytest
import operator
import numpy as np
import depthtensor as dt

@pytest.mark.parametrize(
    "op",
    [
        operator.eq,  # ==
        operator.ne,  # !=
        operator.gt,  # >
        operator.ge,  # >=
        operator.lt,  # <
        operator.le,  # <=
    ],
)
def test_tensor_comparisons(device, op):
    a_data = [1.0, 2.0, 3.0]
    b_data = [1.0, 3.0, 2.0]

    a = dt.Tensor(a_data, device=device, requires_grad=True)
    b = dt.Tensor(b_data, device=device, requires_grad=True)

    result = op(a, b)
    expected = op(np.array(a_data), np.array(b_data))

    result_data = result.data.get() if hasattr(result.data, "get") else result.data
    np.testing.assert_array_equal(result_data, expected)

    assert result.requires_grad is False
    assert result.grad_fn is None
