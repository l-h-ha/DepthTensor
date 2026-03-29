import importlib.util
import pytest

CUDA_AVAILABLE = importlib.util.find_spec("cupy") is not None


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    If a test function has an argument named 'device',
    run it once for 'cpu' and (if available) once for 'cuda'.
    """
    if "device" in metafunc.fixturenames:
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        metafunc.parametrize("device", devices)
