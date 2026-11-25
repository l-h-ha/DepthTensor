import numpy as np
import pytest

import depthtensor as dt

@pytest.mark.parametrize("create_scalar", [
    pytest.param(lambda: dt.Tensor([1.0])[0], id="index"),
    pytest.param(lambda: dt.Tensor([1.0]).sum(), id="sum"),
    pytest.param(lambda: dt.add(1.0, 2.0), id="add"),
    pytest.param(lambda: dt.mean(dt.Tensor([1.0, 2.0]), in_place=True), id="inplace-mean"),
])
def test_cpu_scalar_results_use_writable_array_storage(create_scalar):
    scalar = create_scalar()

    assert isinstance(scalar.data, np.ndarray)
    assert scalar.shape == ()

    scalar[...] = 7.0
    assert scalar.item() == 7.0

    result = dt.add(2.0, 3.0, out=scalar.data)
    assert result.data is scalar.data
    assert scalar.item() == 5.0

def test_indexed_assignment_accepts_tensor_values(device, assert_allclose):
    x = dt.Tensor([1.0, 2.0, 3.0], device=device)

    x[:2] = dt.Tensor([10.0, 20.0], device=device)
    x[2] = dt.Tensor(4.0, device=device)

    assert_allclose(x, [10.0, 20.0, 4.0])
