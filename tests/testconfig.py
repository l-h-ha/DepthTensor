import pytest

CUDA_AVAILABLE = False
try:
    import cupy

    GPU_AVAILABLE = True
except (ImportError, Exception):
    pass


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    If a test function has an argument named 'device',
    run it once for 'cpu' and (if available) once for 'cuda'.
    """
    if "device" in metafunc.fixturenames:
        devices = ["cpu"]
        if GPU_AVAILABLE:
            devices.append("cuda")
        metafunc.parametrize("device", devices)
