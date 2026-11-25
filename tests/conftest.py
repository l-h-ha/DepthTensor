import importlib.util
import pytest
import numpy as np
import depthtensor as dt

CUDA_AVAILABLE = importlib.util.find_spec("cupy") is not None

def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    Run device tests on CPU. Add CUDA if CuPy is available.
    """
    if "device" in metafunc.fixturenames:
        devices = ["cpu"]

        if CUDA_AVAILABLE:
            devices.append("cuda")

        metafunc.parametrize("device", devices)

@pytest.fixture
def assert_allclose():
    """
    Compare Tensor, NumPy, or CuPy values on CPU.

    Returns
    -------
    callable
        Function that checks values within tolerance.
    """
    def _to_numpy(array):
        if hasattr(array, "get"):
            return array.get()

        if isinstance(array, dt.Tensor):
            return _to_numpy(array.data)

        return np.asarray(array)

    def _assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, err_msg=""):
        actual_np = _to_numpy(actual)
        expected_np = _to_numpy(expected)

        np.testing.assert_allclose(
            actual_np,
            expected_np,
            rtol=rtol,
            atol=atol,
            err_msg=err_msg or "Values differ by more than the tolerance",
        )

    return _assert_allclose
